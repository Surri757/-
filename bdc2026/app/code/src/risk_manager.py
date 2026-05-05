"""
风控层 — 以损定仓 + 止损/止盈 + 保护失败处理
"""
from dataclasses import dataclass, field
from typing import Dict, Optional
from datetime import datetime


@dataclass
class RiskPosition:
    """风控持仓记录"""
    stock_id: str
    entry_price: float
    stop_loss_price: float
    take_profit_price: float
    weight: float          # 资金占比
    entry_date: str
    max_loss: float = 0.0  # 最大可能亏损 (金额)
    status: str = 'active'  # active / stopped_out / take_profit / force_closed


class RiskManager:
    """以损定仓风控管理器

    核心逻辑:
    - 先定最大可承受亏损, 再反推仓位大小
    - 仓位 = max_loss_per_position / (入场价 - 止损价) / 入场价
    - 单只股票持仓唯一
    - 保护失败 → 强制平仓记录
    """

    def __init__(self,
                 total_capital: float = 1.0,
                 max_loss_per_position: float = 0.02,
                 max_total_exposure: float = 1.0,
                 single_stock_only: bool = True):
        self.total_capital = total_capital
        self.max_loss_per_position = max_loss_per_position  # 每笔最多亏总资金2%
        self.max_total_exposure = max_total_exposure        # 总仓位上限
        self.single_stock_only = single_stock_only
        self.positions: Dict[str, RiskPosition] = {}
        self.risk_events: list = []  # 风险事件日志

    def compute_position_size(self, signal) -> float:
        """以损定仓: 根据最大可接受亏损计算仓位

        position_size = max_loss_pct / (stop_loss_pct)
        """
        entry = signal.entry_price
        stop = signal.stop_loss_price
        if stop >= entry or entry <= 0:
            return 0.0

        # 每股亏损比例
        loss_pct = (entry - stop) / entry

        # 可买入仓位 = 最大可接受亏损比例 / 每股价亏损比例
        position_size = self.max_loss_per_position / loss_pct

        # 信号置信度调整: 高置信 → 加仓, 低置信 → 减仓
        confidence_mult = 0.5 + 0.5 * signal.confidence
        position_size *= confidence_mult

        # 不超过总仓位上限
        position_size = min(position_size, self.max_total_exposure)

        # 不超过剩余可用仓位
        used = sum(p.weight for p in self.positions.values())
        remaining = self.max_total_exposure - used
        position_size = min(position_size, max(remaining, 0.0))

        return max(0.0, position_size)

    def check_stop_loss(self, stock_id: str, current_price: float) -> bool:
        """检查是否触发止损"""
        pos = self.positions.get(stock_id)
        if pos is None or pos.status != 'active':
            return False
        if current_price <= pos.stop_loss_price:
            pos.status = 'stopped_out'
            self.risk_events.append({
                'type': 'stop_loss',
                'stock_id': stock_id,
                'price': current_price,
                'stop_price': pos.stop_loss_price,
                'time': datetime.now().isoformat(),
            })
            return True
        return False

    def check_take_profit(self, stock_id: str, current_price: float) -> bool:
        """检查是否触发止盈"""
        pos = self.positions.get(stock_id)
        if pos is None or pos.status != 'active':
            return False
        if current_price >= pos.take_profit_price:
            pos.status = 'take_profit'
            self.risk_events.append({
                'type': 'take_profit',
                'stock_id': stock_id,
                'price': current_price,
                'tp_price': pos.take_profit_price,
                'time': datetime.now().isoformat(),
            })
            return True
        return False

    def register_position(self, stock_id: str, position: RiskPosition):
        """登记持仓 (单只股票唯一检查)"""
        if self.single_stock_only and stock_id in self.positions:
            existing = self.positions[stock_id]
            if existing.status == 'active':
                raise ValueError(f"Position already exists for {stock_id}, close first")
        self.positions[stock_id] = position

    def force_close(self, stock_id: str, reason: str) -> Optional[RiskPosition]:
        """强制平仓 (保护失败等异常场景)"""
        pos = self.positions.pop(stock_id, None)
        if pos:
            pos.status = 'force_closed'
            self.risk_events.append({
                'type': 'force_close',
                'stock_id': stock_id,
                'reason': reason,
                'entry_price': pos.entry_price,
                'time': datetime.now().isoformat(),
            })
        return pos

    def get_active_positions(self) -> Dict[str, RiskPosition]:
        """获取活跃持仓"""
        return {k: v for k, v in self.positions.items() if v.status == 'active'}

    def get_risk_summary(self) -> dict:
        """风险汇总"""
        active = self.get_active_positions()
        return {
            'n_active': len(active),
            'total_exposure': sum(p.weight for p in active.values()),
            'max_potential_loss': sum(p.max_loss for p in active.values()),
            'risk_events_count': len(self.risk_events),
            'events_by_type': {
                t: sum(1 for e in self.risk_events if e['type'] == t)
                for t in set(e['type'] for e in self.risk_events)
            },
        }
