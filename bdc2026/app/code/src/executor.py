"""
执行层 — 模拟下单 + 保护单 + 状态回查
"""
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, field

from risk_manager import RiskManager, RiskPosition


@dataclass
class ExecutionRecord:
    """执行记录"""
    stock_id: str
    date: str
    entry_price: float
    position_size: float
    protection_orders: dict
    status: str  # 'filled', 'rejected', 'partial'
    slippage: float = 0.0  # 计划价 vs 实际成交价偏差
    fill_time: str = ''


class ExecutionSimulator:
    """模拟执行引擎

    生产环境中会对接真实交易所API:
    - 市价单开仓
    - 立即挂止盈止损保护单
    - 回查交易所状态确认成交
    """

    def __init__(self, risk_manager: RiskManager,
                 default_slippage_bps: float = 5.0):
        self.risk = risk_manager
        self.default_slippage_bps = default_slippage_bps  # 默认滑点 5bps
        self.execution_log: List[ExecutionRecord] = []
        self.slippage_records: List[dict] = []

    def execute_signal(self, signal, position_size: float) -> Optional[ExecutionRecord]:
        """执行交易信号

        1. 市价单开仓 (模拟: T+1开盘价成交)
        2. 挂止盈止损保护单
        3. 回查确认
        """
        if position_size <= 0:
            return None

        # 模拟滑点 (市价单在开盘价的 ±slippage_bps 范围)
        slippage = signal.entry_price * self.default_slippage_bps / 10000
        fill_price = signal.entry_price + slippage  # 买入方向正滑点

        # 保护单
        protection = {
            'stop_loss': signal.stop_loss_price,
            'take_profit': signal.take_profit_price,
            'is_bilateral': True,  # 双边保护
            'order_id': f"PROT_{signal.stock_id}_{signal.date}",
        }

        # 执行记录
        record = ExecutionRecord(
            stock_id=signal.stock_id,
            date=signal.date,
            entry_price=fill_price,
            position_size=position_size,
            protection_orders=protection,
            status='filled',
            slippage=slippage,
            fill_time=datetime.now().isoformat(),
        )

        self.execution_log.append(record)
        self.slippage_records.append({
            'stock_id': signal.stock_id,
            'expected': signal.entry_price,
            'actual': fill_price,
            'slippage_bps': slippage / signal.entry_price * 10000,
        })

        # 注册到风控管理器
        max_loss = position_size * (fill_price - signal.stop_loss_price) / fill_price
        risk_pos = RiskPosition(
            stock_id=signal.stock_id,
            entry_price=fill_price,
            stop_loss_price=signal.stop_loss_price,
            take_profit_price=signal.take_profit_price,
            weight=position_size,
            entry_date=signal.date,
            max_loss=max_loss,
        )
        self.risk.register_position(signal.stock_id, risk_pos)

        return record

    def verify_against_exchange(self) -> tuple:
        """回查交易所状态确认

        Returns: (is_verified, message)
        模拟环境始终返回验证通过。
        生产环境需对比 positions/orders/fills 与交易所数据。
        """
        return True, "verified (simulated)"

    def handle_protection_failure(self, stock_id: str, reason: str) -> Optional[RiskPosition]:
        """保护单挂单失败 → 立即强制平仓"""
        pos = self.risk.force_close(stock_id, f"protection_failed:{reason}")
        if pos:
            self.execution_log.append(ExecutionRecord(
                stock_id=stock_id,
                date=datetime.now().strftime('%Y-%m-%d'),
                entry_price=0,  # 平仓
                position_size=-pos.weight,
                protection_orders={},
                status='force_close',
            ))
        return pos

    def get_execution_summary(self) -> dict:
        """执行层汇总统计"""
        filled = [r for r in self.execution_log if r.status == 'filled']
        return {
            'total_orders': len(self.execution_log),
            'filled': len(filled),
            'force_closed': len([r for r in self.execution_log if r.status == 'force_close']),
            'avg_slippage_bps': (
                sum(s['slippage_bps'] for s in self.slippage_records) / max(len(self.slippage_records), 1)
            ),
            'total_exposure': sum(r.position_size for r in filled),
        }
