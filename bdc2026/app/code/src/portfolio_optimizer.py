"""
Portfolio Optimizer — 组合权重优化
替代贪婪填仓，使用 scipy 约束优化求解最优权重分配。
"""
import numpy as np
from scipy.optimize import minimize


class PortfolioOptimizer:
    """约束组合优化器

    三种方法:
    - max_sharpe: 最大化 w·μ / sqrt(w·Σ·w)
    - min_variance: 最小化 w·Σ·w
    - max_return_risk_budget: 最大化 w·μ, 约束 w·Σ·w ≤ risk_budget

    约束: w_i ≥ 0, sum(w) = max_weight_sum (满仓), 最终保留 top max_positions 只
    """

    def __init__(self):
        pass

    # ── 公共接口 ──

    def optimize(self, mu, Sigma, method='max_sharpe', max_weight_sum=1.0,
                 max_single=1.0, max_positions=5, risk_budget=0.01):
        """优化组合权重

        Args:
            mu: 预期收益向量 (n_candidates,)
            Sigma: 协方差矩阵 (n_candidates, n_candidates)
            method: 'max_sharpe' | 'min_variance' | 'max_return_risk_budget'
            max_weight_sum: 权重和上限
            max_single: 单只权重上限
            max_positions: 最大持仓数
            risk_budget: max_return_risk_budget 的风险预算

        Returns:
            weights: 优化后权重 (长度 = max_positions, 非零部分)
            stock_indices: 对应原始候选列表中的索引
            info: 诊断信息 dict
        """
        n = len(mu)
        if n == 0:
            return np.array([]), np.array([], dtype=int), {'method': method, 'status': 'no_candidates'}

        # 确保 Sigma 正定
        Sigma = self._ensure_psd(Sigma)

        if method == 'max_sharpe':
            w_opt = self._optimize_max_sharpe(mu, Sigma, max_weight_sum, max_single)
        elif method == 'min_variance':
            w_opt = self._optimize_min_variance(Sigma, max_weight_sum, max_single)
        elif method == 'max_return_risk_budget':
            w_opt = self._optimize_max_return_risk_budget(
                mu, Sigma, max_weight_sum, max_single, risk_budget
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        # 后处理：零化小权重 + 保留 top max_positions
        w_opt = self._zero_small_weights(w_opt, threshold=1e-4)
        w_final, indices = self._select_top_k(w_opt, k=max_positions)

        # 重归一化到 max_weight_sum (强制满仓)
        if w_final.sum() > 0:
            w_final = w_final / w_final.sum() * max_weight_sum

        info = {
            'method': method,
            'n_candidates': n,
            'n_selected': len(w_final),
            'weight_sum': float(w_final.sum()),
            'max_single_weight': float(w_final.max()) if len(w_final) > 0 else 0,
            'status': 'ok',
        }

        return w_final, indices, info

    # ── 优化核心 ──

    def _optimize_max_sharpe(self, mu, Sigma, max_weight_sum, max_single):
        """最大化 Sharpe ratio: w·μ / sqrt(w·Σ·w)"""
        n = len(mu)
        # 初始值: 等权
        w0 = np.ones(n) / n * min(max_weight_sum / n, max_single)

        # 约束: w_i ≥ 0, w_i ≤ max_single, sum(w) = max_weight_sum
        bounds = [(0, max_single) for _ in range(n)]
        constraints = {'type': 'eq', 'fun': lambda w: max_weight_sum - w.sum()}

        # 用 minimize 最大化 Sharpe (minimize negative Sharpe on non-zero weights)
        def neg_sharpe(w):
            port_return = w @ mu
            port_risk = np.sqrt(w @ Sigma @ w + 1e-10)
            return -port_return / port_risk

        result = minimize(
            neg_sharpe, w0, method='SLSQP', bounds=bounds,
            constraints=constraints,
            options={'maxiter': 500, 'ftol': 1e-10}
        )

        if not result.success:
            # 回退到等权
            return w0

        w = np.maximum(result.x, 0.0)
        return w

    def _optimize_min_variance(self, Sigma, max_weight_sum, max_single):
        """最小化组合方差"""
        n = Sigma.shape[0]
        w0 = np.ones(n) / n * min(max_weight_sum / n, max_single)

        bounds = [(0, max_single) for _ in range(n)]
        constraints = {'type': 'eq', 'fun': lambda w: max_weight_sum - w.sum()}

        def portfolio_variance(w):
            return w @ Sigma @ w

        result = minimize(
            portfolio_variance, w0, method='SLSQP', bounds=bounds,
            constraints=constraints,
            options={'maxiter': 500, 'ftol': 1e-10}
        )

        if not result.success:
            return w0

        return np.maximum(result.x, 0.0)

    def _optimize_max_return_risk_budget(self, mu, Sigma, max_weight_sum, max_single, risk_budget):
        """最大化收益，约束方差不超过 risk_budget"""
        n = len(mu)
        w0 = np.ones(n) / n * min(max_weight_sum / n, max_single)

        bounds = [(0, max_single) for _ in range(n)]
        constraints = [
            {'type': 'eq', 'fun': lambda w: max_weight_sum - w.sum()},
            {'type': 'ineq', 'fun': lambda w: risk_budget - w @ Sigma @ w},
        ]

        def neg_return(w):
            return -(w @ mu)

        result = minimize(
            neg_return, w0, method='SLSQP', bounds=bounds,
            constraints=constraints,
            options={'maxiter': 500, 'ftol': 1e-10}
        )

        if not result.success:
            return w0

        return np.maximum(result.x, 0.0)

    # ── 后处理 ──

    @staticmethod
    def _zero_small_weights(weights, threshold=1e-4):
        w = np.asarray(weights).copy()
        w[w < threshold] = 0.0
        return w

    @staticmethod
    def _select_top_k(weights, k):
        """保留 top-k 非零权重"""
        nz_idx = np.where(weights > 0)[0]
        if len(nz_idx) <= k:
            return weights[nz_idx], nz_idx
        # 取最大的 k 个
        top_k_idx = nz_idx[np.argsort(weights[nz_idx])[-k:]]
        w_k = weights[top_k_idx]
        return w_k, top_k_idx

    # ── 协方差估计 ──

    @staticmethod
    def estimate_covariance_from_returns(price_df, stock_ids, lookback=60):
        """从历史价格估计协方差矩阵

        Args:
            price_df: DataFrame, 包含 stock_id, date, close 列
            stock_ids: 候选股票代码列表
            lookback: 回看天数

        Returns:
            Sigma: (n, n) 协方差矩阵
        """
        n = len(stock_ids)
        if n <= 1:
            return np.eye(n) * 0.01

        # 为每只股票计算日收益率
        returns_list = []
        valid_stocks = []
        for sid in stock_ids:
            sub = price_df[price_df['stock_id'].astype(str).str.zfill(6) == sid]
            if len(sub) < lookback:
                continue
            sub = sub.sort_values('date')
            close = sub['close'].values[-lookback:]
            ret = np.diff(np.log(close + 1e-10))
            if len(ret) >= 20:
                returns_list.append(ret[-min(lookback - 1, len(ret)):])
                valid_stocks.append(sid)

        if len(returns_list) < 2:
            return np.eye(n) * 0.01

        # 对齐长度
        min_len = min(len(r) for r in returns_list)
        aligned = np.array([r[-min_len:] for r in returns_list])

        # 协方差矩阵 + 收缩估计 (Ledoit-Wolf 简化版)
        S = np.cov(aligned)
        # 收缩到对角目标
        target = np.diag(np.diag(S))
        shrinkage = 0.2
        Sigma_shrunk = (1 - shrinkage) * S + shrinkage * target

        # 如果部分股票无数据，填充
        if len(valid_stocks) < n:
            full_Sigma = np.eye(n) * 0.01
            idx_map = {sid: i for i, sid in enumerate(stock_ids)}
            for i, si in enumerate(valid_stocks):
                for j, sj in enumerate(valid_stocks):
                    full_Sigma[idx_map[si], idx_map[sj]] = Sigma_shrunk[i, j]
            return full_Sigma

        return Sigma_shrunk

    @staticmethod
    def estimate_covariance_from_signals(signals):
        """从模型分歧（model_std）估计对角协方差 + 常数相关"""
        n = len(signals)
        if n <= 1:
            return np.eye(n) * 0.01

        stds = np.array([s.model_std for s in signals])
        stds = np.clip(stds, 0.005, 0.1)
        # 常数相关性模型：同行业 0.3，不同行业 0.1
        corr = np.full((n, n), 0.1)
        np.fill_diagonal(corr, 1.0)
        # 用 volatility_cluster 分组 (同簇高相关)
        for i in range(n):
            for j in range(i + 1, n):
                if getattr(signals[i], 'volatility_cluster', None) == getattr(signals[j], 'volatility_cluster', None):
                    corr[i, j] = corr[j, i] = 0.25

        Sigma = np.outer(stds, stds) * corr
        return Sigma

    @staticmethod
    def _ensure_psd(Sigma):
        """确保协方差矩阵半正定"""
        eigvals, eigvecs = np.linalg.eigh(Sigma)
        eigvals = np.maximum(eigvals, 1e-8)
        return eigvecs @ np.diag(eigvals) @ eigvecs.T
