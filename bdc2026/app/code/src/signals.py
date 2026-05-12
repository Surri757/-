"""
信号层 — 将模型预测转换为结构化交易信号
"""
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
import pandas as pd


@dataclass
class StockSignal:
    """单只股票的交易信号"""
    stock_id: str
    date: str
    direction: int              # 1=做多, 0=持有, -1=做空
    entry_price: float          # 预期T+1开盘价 (买入价)
    predicted_return: float     # 预测5日收益率
    stop_loss_price: float      # 止损价
    take_profit_price: float    # 止盈价
    confidence: float           # 0~1 模型置信度
    model_std: float            # 跨模型预测标准差
    volatility_cluster: int = 1  # 波动率聚类 0低/1中/2高

    def __post_init__(self):
        if self.stop_loss_price >= self.entry_price:
            self.stop_loss_price = self.entry_price * 0.93
        if self.take_profit_price <= self.entry_price:
            self.take_profit_price = self.entry_price * (1 + max(self.predicted_return, 0.05))


class SignalGenerator:
    """信号生成器：特征工程 + 模型预测 + 信号增强 (含横截面增强)"""

    def __init__(self, predictor, feature_eng, industry_df=None, macro_df=None,
                 seq_len=60, sl_atr_multiplier=2.0, stock_industry_map=None):
        self.predictor = predictor
        self.fe = feature_eng
        self.industry_df = industry_df
        self.macro_df = macro_df
        self.seq_len = seq_len
        self.sl_atr_multiplier = sl_atr_multiplier
        self.stock_industry_map = stock_industry_map or {}

    def generate_all(self, df: pd.DataFrame, stock_ids: List[str]) -> List[StockSignal]:
        """为所有股票生成信号 (含横截面增强)"""
        signals = []
        for stock_id in stock_ids:
            signal = self.generate_one(df, stock_id)
            if signal is not None:
                signals.append(signal)

        # 横截面增强: 让信号包含"相对强弱"信息
        if len(signals) >= 5:
            signals = self._apply_cross_sectional_enhancement(signals)

        return signals

    def generate_one(self, df: pd.DataFrame, stock_id: str) -> Optional[StockSignal]:
        """为单只股票生成信号"""
        stock_df = df[df['stock_id'].astype(str).str.zfill(6) == stock_id].copy()
        if len(stock_df) < self.seq_len:
            return None

        stock_df = stock_df.sort_values('date').set_index('date')

        # 特征工程
        features = self.fe.build_all_features(stock_df, self.industry_df, self.macro_df)
        features = self.fe.remove_outliers(features)

        last_features = features.iloc[-self.seq_len:].fillna(0)
        X_raw = last_features.values

        # 波动率聚类
        from train import compute_volatility_cluster
        cluster_id = compute_volatility_cluster(stock_df['close']) if 'close' in stock_df.columns else 1
        cluster_col = np.full((len(X_raw), 1), cluster_id, dtype=np.float32)
        X = np.column_stack([X_raw, cluster_col])

        # 预测
        pred_return, pred_std = self.predictor.predict(
            X, historical_prices=stock_df['close'].values if 'close' in stock_df.columns else None)

        # 信号增强
        latest = stock_df.iloc[-1]
        entry_price = float(latest.get('open', latest.get('close', 10)))

        # ATR-based止损 (自适应波动率)
        atr = self._compute_atr(stock_df)
        sl_distance = max(atr * self.sl_atr_multiplier, entry_price * 0.03)
        stop_loss = entry_price - sl_distance

        # 止盈基于预测收益
        take_profit = entry_price * (1 + max(pred_return, 0.03))

        # 置信度 (基于预测一致性)
        confidence = float(np.clip(1.0 - pred_std / (abs(pred_return) + 0.02), 0.0, 1.0))

        direction = 1 if pred_return > 0.005 else (0 if pred_return > -0.005 else -1)

        latest_date = str(stock_df.index[-1].date()) if hasattr(stock_df.index[-1], 'date') else str(stock_df.index[-1])

        return StockSignal(
            stock_id=stock_id,
            date=latest_date,
            direction=direction,
            entry_price=entry_price,
            predicted_return=float(pred_return),
            stop_loss_price=float(stop_loss),
            take_profit_price=float(take_profit),
            confidence=confidence,
            model_std=float(pred_std),
            volatility_cluster=cluster_id,
        )

    def _apply_cross_sectional_enhancement(self, signals: List[StockSignal]) -> List[StockSignal]:
        """横截面增强: 同一天所有股票之间做相对强弱排名

        核心逻辑:
        - 截面 z-score: 该股票在同日所有股票中的相对位置
        - 行业 z-score: 该股票在同行业内的相对位置
        - 将相对排名注入 predicted_return 和 confidence
        - 绝对预测强 + 相对排名高 = 最强信号
        """
        returns = np.array([s.predicted_return for s in signals])
        mean_ret = float(np.mean(returns))
        std_ret = float(np.std(returns))
        if std_ret < 1e-6:
            return signals

        # ── 截面 z-score ──
        cs_zscores = (returns - mean_ret) / std_ret

        # ── 行业相对 z-score ──
        ind_zscores = np.zeros(len(signals))
        if self.stock_industry_map:
            ind_groups = {}
            for i, s in enumerate(signals):
                ind = self.stock_industry_map.get(s.stock_id, '其他')
                ind_groups.setdefault(ind, []).append(i)
            for ind, indices in ind_groups.items():
                if len(indices) >= 3:
                    ind_rets = returns[indices]
                    ind_std = float(np.std(ind_rets))
                    if ind_std > 1e-6:
                        ind_zscores[indices] = (ind_rets - float(np.mean(ind_rets))) / ind_std

        # ── 增强: 将相对排名注入 predicted_return ──
        # 混合权重: 60% 绝对预测 + 40% 截面相对位置
        # 截面位置 = mean + cs_z * std (该股票在截面分布中的"合理"预测值)
        cross_sectional_value = mean_ret + cs_zscores * std_ret
        for i, s in enumerate(signals):
            # 行业相对也参与混合
            ind_weight = 0.15 if abs(ind_zscores[i]) > 0.5 else 0.0
            cs_weight = 0.40 - ind_weight

            enhanced_ret = ((1.0 - cs_weight - ind_weight) * s.predicted_return
                            + cs_weight * cross_sectional_value[i]
                            + ind_weight * (mean_ret + ind_zscores[i] * std_ret))

            s.predicted_return = float(enhanced_ret)

            # 置信度增强: 截面排名前 20% + 行业排名前 30% 的信号更可信
            cs_bonus = max(0.0, cs_zscores[i]) * 0.15
            ind_bonus = max(0.0, ind_zscores[i]) * 0.08
            s.confidence = float(np.clip(s.confidence + cs_bonus + ind_bonus, 0.0, 1.0))

        top_n = min(5, len(signals))
        top_cs = int(np.sum(cs_zscores >= 1.0))
        top_ind = int(np.sum(ind_zscores >= 1.0))
        print(f"[横截面增强] {len(signals)}只 → 截面z≥1: {top_cs}只, 行业z≥1: {top_ind}只 | "
              f"均值={mean_ret:.3%} σ={std_ret:.3%} | 前{top_n}均值={np.mean(sorted(returns)[-top_n:]):.3%}")

        return signals

    @staticmethod
    def _compute_atr(stock_df, period=14):
        """计算ATR用于自适应止损"""
        if len(stock_df) < period:
            return float(stock_df['close'].iloc[-1] * 0.02)
        high = stock_df['high']
        low = stock_df['low']
        close = stock_df['close']
        hl = high - low
        hc = (high - close.shift()).abs()
        lc = (low - close.shift()).abs()
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        return float(tr.tail(period).mean())
