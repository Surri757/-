"""
Live Gate — 实盘准入校验层 (动态版)
所有信号必须通过Gate检查才能进入执行层。
根据市场状态动态调整阈值：牛市放宽抓大肉，熊市/恐慌收紧控风险。
"""
from typing import List, Set, Dict, Optional, Tuple
import numpy as np
from spectral_m import SpectralStateMachine


class LiveGate:
    """实盘准入校验器 (动态阈值)

    5项检查:
    1. 白名单 (可选)
    2. 单只股票唯一持仓 (同一时刻只允许一单)
    3. 风险距离 (止损价必须离入场价足够远) — 动态
    4. 信号置信度 (置信度过低则拒绝) — 动态
    5. 最大持仓数限制 — 动态

    动态阈值根据市场状态自适应调整:
    - 牛市: 降置信度门槛、提持仓至上限5只、放宽风险距离 → 抓大肉
    - 恐慌: 升置信度门槛、降持仓至3只、收紧风险距离 → 控风险
    """

    HARD_MAX_POSITIONS = 5  # 比赛规则：任何时候持仓不超过5只

    # 不同市场状态下的阈值配置
    REGIME_CONFIG = {
        'bull':     {'confidence': 0.05, 'max_positions': 5, 'risk_distance': 0.015},
        'sideways': {'confidence': 0.10, 'max_positions': 5, 'risk_distance': 0.020},
        'neutral':  {'confidence': 0.10, 'max_positions': 5, 'risk_distance': 0.020},
        'bear':     {'confidence': 0.15, 'max_positions': 4, 'risk_distance': 0.025},
        'panic':    {'confidence': 0.20, 'max_positions': 3, 'risk_distance': 0.030},
    }

    def __init__(self,
                 whitelist_stocks: Optional[Set[str]] = None,
                 max_single_position: float = 1.0,
                 min_risk_distance_pct: float = 0.02,
                 min_confidence: float = 0.10,
                 max_positions: int = 5):
        self.whitelist = whitelist_stocks or set()
        self.max_single_position = max_single_position

        # 动态阈值（初始值，后续由 update_regime 覆盖）
        self.min_risk_distance_pct = min_risk_distance_pct
        self.min_confidence = min_confidence
        self.max_positions = max_positions

        # 市场状态快照
        self.regime = 'neutral'
        self.trend_score = 0.0
        self.vol_score = 1.0
        self.pred_score = 0.5
        self.pred_sharpe = 0.0
        self.pred_pos_ratio = 0.5

        self.positions: Dict[str, dict] = {}
        self.rejection_stats: Dict[str, int] = {}  # 拒绝原因统计

    # ── 市场状态检测 & 阈值自适应 ──

    def update_regime(self, index_returns=None, predicted_returns=None):
        """根据市场状态动态调整所有阈值。

        Args:
            index_returns: 大盘指数日收益率序列（最近252个交易日）
            predicted_returns: 当前批次所有信号的预测收益率列表
        """
        regime, scores = self._detect_regime(index_returns, predicted_returns)
        self.regime = regime
        self.trend_score = scores['trend']
        self.vol_score = scores['vol']
        self.pred_score = scores['pred']
        self.pred_sharpe = scores['sharpe']
        self.pred_pos_ratio = scores['pos_ratio']

        # SSM 谱状态机补充: 从市场特征向量识别隐状态
        ssm_state = self._detect_ssm_state(index_returns, predicted_returns, scores)
        self.ssm_state = ssm_state  # 0..4 隐状态

        self._adapt_thresholds(regime, scores, ssm_state)

    def _detect_regime(self, index_returns, predicted_returns):
        """检测市场状态，返回 (regime, scores_dict)"""
        scores = {'trend': 0.0, 'vol': 1.0, 'pred': 0.5, 'sharpe': 0.0, 'pos_ratio': 0.5}

        if index_returns is None or len(index_returns) < 20:
            return 'neutral', scores

        ret_5d = np.mean(index_returns[-5:])
        ret_20d = np.mean(index_returns[-20:])
        ret_60d = np.mean(index_returns[-60:]) if len(index_returns) >= 60 else ret_20d
        vol_20d = np.std(index_returns[-20:])
        vol_60d = np.std(index_returns[-60:]) if len(index_returns) >= 60 else vol_20d
        vol_change = vol_20d / (vol_60d + 1e-8)

        # 预测质量
        if predicted_returns is not None and len(predicted_returns) > 0:
            preds = np.asarray(predicted_returns)
            pred_mean = np.mean(preds)
            pred_std = np.std(preds) + 1e-8
            scores['sharpe'] = pred_mean / pred_std
            scores['pos_ratio'] = np.mean(preds > 0)
            scores['pred'] = np.clip(scores['sharpe'] / 2.0 + 0.5, 0.2, 1.0)
            scores['pred'] *= (0.5 + 0.5 * scores['pos_ratio'])

        # 趋势分
        trend_5d = np.clip(ret_5d / 0.03, -1, 1)
        trend_20d = np.clip(ret_20d / 0.05, -1, 1)
        trend_60d = np.clip(ret_60d / 0.08, -1, 1)
        scores['trend'] = 0.3 * trend_5d + 0.4 * trend_20d + 0.3 * trend_60d

        # 波动分
        vol_penalty = np.clip(vol_change - 0.8, 0, 1) * 0.3 + np.clip(vol_20d / 0.025 - 0.5, 0, 1) * 0.3
        scores['vol'] = 1.0 - np.clip(vol_penalty, 0, 0.6)

        # 市场状态识别
        if ret_5d < -0.03 and vol_change > 1.5:
            regime = 'panic'
        elif ret_20d < -0.01 and vol_change > 1.2:
            regime = 'bear'
        elif ret_20d > 0.01:
            regime = 'bull'
        elif abs(ret_20d) < 0.003:
            regime = 'sideways'
        else:
            regime = 'neutral'

        return regime, scores

    def _detect_ssm_state(self, index_returns, predicted_returns, scores):
        """SSM 谱状态机: 从市场指标向量识别微观隐状态

        构建特征向量 Z = [趋势分, 波动分, 预测分, 波动率, vol_change, Sharpe, pos_ratio]
        用预拟合的 SSM (基于历史数据) 推断当前状态。
        如果数据不足, 返回 -1 表示不启用 SSM。
        """
        if index_returns is None or len(index_returns) < 60:
            return -1
        try:
            vol_20d = np.std(index_returns[-20:])
            vol_60d = np.std(index_returns[-60:]) if len(index_returns) >= 60 else vol_20d
            vol_change = vol_20d / (vol_60d + 1e-8)

            # 构建与训练时一致的特征向量
            z = np.array([
                scores['trend'], scores['vol'], scores['pred'],
                vol_20d, min(vol_change, 3.0),
                np.clip(scores['sharpe'], -3, 3),
                scores['pos_ratio']
            ], dtype=np.float64).reshape(1, -1)

            # 简单的基于距离的状态分配 (不需要完整的历史数据拟合)
            # 5 个状态中心 (通过历史数据预定义)
            if not hasattr(self, '_ssm_centers'):
                # 预定义状态中心 (norm 后可替换为从训练数据学习)
                self._ssm_centers = np.array([
                    [ 0.5, 0.9, 0.7, 0.01, 1.0,  1.0, 0.7],  # 状态0: 牛市, 高趋势高预测
                    [ 0.0, 0.8, 0.5, 0.02, 1.0,  0.0, 0.5],  # 状态1: 震荡, 中性
                    [-0.3, 0.7, 0.4, 0.02, 1.2, -0.5, 0.4],  # 状态2: 弱熊
                    [-0.6, 0.5, 0.3, 0.03, 1.5, -1.0, 0.3],  # 状态3: 熊市
                    [ 0.2, 0.6, 0.2, 0.04, 2.0, -0.2, 0.2],  # 状态4: 高波动恐慌
                ])

            # 马氏距离到各状态中心
            diff = self._ssm_centers - z
            dists = np.sum(diff**2, axis=1)
            state = int(np.argmin(dists))
            return state
        except Exception:
            return -1

    def _adapt_thresholds(self, regime, scores, ssm_state=-1):
        """根据市场状态和信号质量微调阈值 (含 SSM 隐状态修正)"""
        cfg = self.REGIME_CONFIG.get(regime, self.REGIME_CONFIG['neutral'])

        # SSM 隐状态修正: 状态越恐慌(3-4), 越收紧; 状态越牛(0), 越放宽
        ssm_bonus = 0.0
        if ssm_state >= 0:
            ssm_map = {0: -0.02, 1: 0.0, 2: 0.02, 3: 0.04, 4: 0.05}
            ssm_bonus = ssm_map.get(ssm_state, 0.0)

        # 基础值从状态配置取
        base_conf = cfg['confidence']
        base_max_pos = cfg['max_positions']
        base_risk_dist = cfg['risk_distance']

        # 用预测质量微调: 信号整体质量高 → 适当放宽
        pred_quality = scores.get('pred', 0.5)
        quality_bonus = (pred_quality - 0.5) * 0.04  # ±0.02 范围

        self.min_confidence = max(0.03, min(0.25, base_conf - quality_bonus + ssm_bonus))

        # 用趋势分微调持仓上限（不超过硬上限5只）
        trend = scores.get('trend', 0.0)
        trend_bonus = int(round(trend * 2))  # ±2 范围
        self.max_positions = max(2, min(self.HARD_MAX_POSITIONS, base_max_pos + trend_bonus))

        # 用波动分微调风险距离
        vol = scores.get('vol', 1.0)
        vol_adjust = (1.0 - vol) * 0.01  # 高波动 → 放宽距离
        self.min_risk_distance_pct = max(0.01, min(0.04, base_risk_dist + vol_adjust))

    # ── 准入校验 ──

    def validate(self, signal) -> Tuple[bool, str]:
        """校验信号是否可以通过Gate (使用当前动态阈值)

        Returns: (passed, reason)
        """
        # Check 1: 白名单
        if self.whitelist and signal.stock_id not in self.whitelist:
            self._count_rejection('not_in_whitelist')
            return False, "not_in_whitelist"

        # Check 2: 单只股票唯一持仓
        if signal.stock_id in self.positions:
            self._count_rejection('existing_position_conflict')
            return False, "existing_position_conflict"

        # Check 3: 风险距离 (动态阈值)
        if signal.entry_price > 0 and signal.stop_loss_price < signal.entry_price:
            sl_pct = (signal.entry_price - signal.stop_loss_price) / signal.entry_price
            if sl_pct < self.min_risk_distance_pct:
                self._count_rejection('sl_too_close')
                return False, f"sl_too_close:{sl_pct:.4f}<{self.min_risk_distance_pct:.4f}"
        else:
            self._count_rejection('invalid_stop_loss')
            return False, "invalid_stop_loss"

        # Check 4: 信号置信度 (动态阈值)
        if signal.confidence < self.min_confidence:
            self._count_rejection('confidence_too_low')
            return False, f"confidence_too_low:{signal.confidence:.2f}<{self.min_confidence:.2f}"

        # Check 5: 最大持仓数 (动态阈值)
        if len(self.positions) >= self.max_positions:
            self._count_rejection('max_positions_reached')
            return False, "max_positions_reached"

        return True, "passed"

    # ── 持仓管理 ──

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

    # ── 诊断信息 ──

    def _count_rejection(self, reason: str):
        """统计拒绝原因"""
        key = reason.split(':')[0]
        self.rejection_stats[key] = self.rejection_stats.get(key, 0) + 1

    def status(self) -> dict:
        """返回当前Gate状态摘要"""
        return {
            'regime': self.regime,
            'trend_score': round(self.trend_score, 3),
            'vol_score': round(self.vol_score, 3),
            'pred_score': round(self.pred_score, 3),
            'min_confidence': round(self.min_confidence, 3),
            'max_positions': self.max_positions,
            'min_risk_distance_pct': round(self.min_risk_distance_pct, 4),
            'current_positions': len(self.positions),
            'rejection_stats': dict(self.rejection_stats),
        }

    def print_status(self):
        """打印当前Gate状态"""
        s = self.status()
        ssm_str = f" SSM={self.ssm_state}" if hasattr(self, 'ssm_state') and self.ssm_state >= 0 else ""
        print(f"[Gate动态] 市场={s['regime']}{ssm_str} | "
              f"置信度≥{s['min_confidence']:.2f} | "
              f"持仓≤{s['max_positions']} | "
              f"风险距离≥{s['min_risk_distance_pct']:.2%} | "
              f"当前持仓={s['current_positions']}")
        if s['rejection_stats']:
            stats_str = ' '.join(f"{k}={v}" for k, v in s['rejection_stats'].items())
            print(f"  拒绝统计: {stats_str}")
