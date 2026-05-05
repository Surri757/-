"""
Live Gate — 实盘准入校验层
所有信号必须通过Gate检查才能进入执行层
"""
from typing import List, Set, Dict, Optional, Tuple


class LiveGate:
    """实盘准入校验器

    5项检查:
    1. 白名单 (可选)
    2. 单只股票唯一持仓 (同一时刻只允许一单)
    3. 风险距离 (止损价必须离入场价足够远)
    4. 信号置信度 (置信度过低则拒绝)
    5. 最大持仓数限制
    """

    def __init__(self,
                 whitelist_stocks: Optional[Set[str]] = None,
                 max_single_position: float = 1.0,
                 min_risk_distance_pct: float = 0.02,
                 min_confidence: float = 0.10,
                 max_positions: int = 5):
        self.whitelist = whitelist_stocks or set()
        self.max_single_position = max_single_position
        self.min_risk_distance_pct = min_risk_distance_pct
        self.min_confidence = min_confidence
        self.max_positions = max_positions
        self.positions: Dict[str, dict] = {}  # stock_id -> position_info

    def validate(self, signal) -> Tuple[bool, str]:
        """
        校验信号是否可以通过Gate
        Returns: (passed, reason)
        """
        # Check 1: 白名单
        if self.whitelist and signal.stock_id not in self.whitelist:
            return False, "not_in_whitelist"

        # Check 2: 单只股票唯一持仓
        if signal.stock_id in self.positions:
            return False, "existing_position_conflict"

        # Check 3: 风险距离 (止损不能太近)
        if signal.entry_price > 0 and signal.stop_loss_price < signal.entry_price:
            sl_pct = (signal.entry_price - signal.stop_loss_price) / signal.entry_price
            if sl_pct < self.min_risk_distance_pct:
                return False, f"sl_too_close:{sl_pct:.4f}"
        else:
            return False, "invalid_stop_loss"

        # Check 4: 信号置信度
        if signal.confidence < self.min_confidence:
            return False, f"confidence_too_low:{signal.confidence:.2f}"

        # Check 5: 最大持仓数
        if len(self.positions) >= self.max_positions:
            return False, "max_positions_reached"

        return True, "passed"

    def register_position(self, stock_id: str, position_info: dict):
        """登记新持仓"""
        self.positions[stock_id] = position_info

    def release_position(self, stock_id: str) -> Optional[dict]:
        """释放持仓"""
        return self.positions.pop(stock_id, None)

    def get_positions(self) -> List[str]:
        """获取当前持仓列表"""
        return list(self.positions.keys())

    def is_position_held(self, stock_id: str) -> bool:
        """检查是否持有某股票"""
        return stock_id in self.positions
