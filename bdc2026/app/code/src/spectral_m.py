"""
SPECTRAL-M 模块 — 从 Kimi 框架提取并适配到现有 pipeline。
HOAT: 高阶自相关特征 | VME: 波动率流形嵌入 | SSM: 谱状态机 | AKRR: 自适应核岭回归
"""
import numpy as np
from scipy import linalg
from scipy.spatial.distance import cdist
from sklearn.cluster import k_means
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class HOATExtractor:
    """高阶自相关张量 — 提取偏度/峰度/交叉矩特征"""
    def __init__(self, window_size=60, n_components=8, estimation_window=30):
        self.window_size = window_size
        self.n_components = n_components
        self.estimation_window = estimation_window
        self.scaler = StandardScaler()
        self._fitted = False

    def _compute_centralized_moments(self, returns_window):
        centered = returns_window - np.mean(returns_window)
        m = len(centered)
        moments = []
        for lag in range(min(5, m)):
            moments.append(np.mean(centered**2) if lag == 0
                          else np.mean(centered[:-lag] * centered[lag:]))
        moments.append(np.mean(centered**3))
        for lag1 in range(1, min(4, m)):
            for lag2 in range(lag1, min(4, m)):
                moments.append(np.mean(centered[:-max(lag1, lag2)] *
                    centered[lag1:-(max(lag1,lag2)-lag1) if max(lag1,lag2)!=lag1 else None] *
                    centered[max(lag1, lag2):]))
        moments.append(np.mean(centered**4))
        cumulants = [np.mean(centered**2), np.mean(centered**3),
                     np.mean(centered**4) - 3 * (np.mean(centered**2)**2)]
        return np.array(moments + cumulants, dtype=np.float32)

    def fit_transform(self, returns):
        n = len(returns)
        min_len = self.window_size + self.estimation_window
        if n < min_len:
            return np.zeros((1, self.n_components), dtype=np.float32), [n - 1]
        raw_features = []
        valid_indices = []
        for t in range(min_len - 1, n):
            window = returns[t - self.window_size + 1:t + 1]
            raw_features.append(self._compute_centralized_moments(window))
            valid_indices.append(t)
        raw_features = np.array(raw_features, dtype=np.float32)
        raw_features = self.scaler.fit_transform(raw_features)
        n_comp = min(self.n_components, raw_features.shape[1])
        pca = PCA(n_components=n_comp, svd_solver='full')
        reduced = pca.fit_transform(raw_features).astype(np.float32)
        self._fitted = True
        return reduced, valid_indices


class VMEmbedder:
    """波动率流形嵌入 — 局部协方差特征分解"""
    def __init__(self, window_size=60, local_window=20, n_components=6):
        self.window_size = window_size
        self.local_window = local_window
        self.n_components = n_components
        self.scaler = StandardScaler()

    def _compute_local_cov_features(self, returns_window_history):
        mu_t = np.mean(returns_window_history, axis=0)
        centered = returns_window_history - mu_t
        Sigma = np.dot(centered.T, centered) / self.local_window
        eigenvalues, eigenvectors = linalg.eigh(Sigma)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = np.maximum(eigenvalues[idx], 1e-10)
        eigenvectors = eigenvectors[:, idx]
        n_comp = min(self.n_components, len(eigenvalues))
        log_eigenvalues = np.log(eigenvalues[:n_comp])
        principal_vectors = eigenvectors[:, :n_comp].flatten()
        return np.concatenate([log_eigenvalues, principal_vectors,
                              [np.trace(Sigma)], [np.linalg.cond(Sigma)]])

    def fit_transform(self, returns):
        n = len(returns)
        min_required = self.window_size + self.local_window
        if n < min_required:
            return np.zeros((1, 6), dtype=np.float32), [n - 1]
        raw_features = []
        valid_indices = []
        for t in range(min_required, n):
            history = []
            for tau in range(min(self.local_window, t - self.window_size)):
                end_idx = t - tau
                start_idx = max(0, end_idx - self.window_size)
                history.append(returns[start_idx:end_idx])
            if len(history) < 3:
                continue
            history = np.array(history)
            raw_features.append(self._compute_local_cov_features(history))
            valid_indices.append(t)
        raw_features = np.array(raw_features, dtype=np.float32)
        raw_features = self.scaler.fit_transform(raw_features)
        n_pca = min(self.n_components, raw_features.shape[1])
        pca = PCA(n_components=n_pca, svd_solver='full')
        return pca.fit_transform(raw_features).astype(np.float32), valid_indices


class SpectralStateMachine:
    """谱分解隐状态机 — 图聚类识别市场微观状态

    比简单阈值 MarketRegime 更精细:
    - 自适应相似度 (余弦 × RBF)
    - 谱聚类 → 4 个隐状态
    - 输出状态概率 (软分配)
    - 状态转移矩阵
    """
    def __init__(self, n_states=5, gamma=2.0, epsilon=None):
        self.n_states = n_states
        self.gamma = gamma
        self.epsilon = epsilon
        self.transition_matrix = None
        self.state_labels = None
        self.state_centers = None

    def _build_adaptive_similarity(self, Z):
        Z_norm = Z / (linalg.norm(Z, axis=1, keepdims=True) + 1e-10)
        cosine_sim = np.abs(np.dot(Z_norm, Z_norm.T)) ** self.gamma
        dist_sq = cdist(Z, Z, 'sqeuclidean')
        if self.epsilon is None:
            k = max(int(np.sqrt(len(Z))), 5)
            sorted_dists = np.sort(dist_sq, axis=1)
            self.epsilon = np.median(sorted_dists[:, 1:k+1]) + 1e-10
        rbf_sim = np.exp(-dist_sq / (2 * self.epsilon))
        A = cosine_sim * rbf_sim
        A = (A + A.T) / 2
        np.fill_diagonal(A, 0)
        return A

    def _normalized_graph_laplacian(self, A):
        D = np.sum(A, axis=1)
        D_inv_sqrt = np.diag(1.0 / np.sqrt(D + 1e-10))
        return np.eye(A.shape[0]) - D_inv_sqrt @ A @ D_inv_sqrt

    def fit_predict(self, Z):
        """谱聚类 → 隐状态标签 + 软概率

        当 N > 5000 时先在随机子集上做谱聚类发现状态结构,
        再将全量数据按最近状态中心分配。避免 N×N 相似度矩阵 OOM。
        """
        n = Z.shape[0]
        if n < self.n_states * 2:
            return np.zeros(n, dtype=int), np.ones((n, self.n_states)) / self.n_states

        max_spec = 5000
        if n > max_spec:
            rng = np.random.RandomState(42)
            sample_idx = rng.choice(n, max_spec, replace=False)
            Z_spec = Z[sample_idx]
        else:
            Z_spec = Z
            sample_idx = np.arange(n)

        # 在子集上做谱聚类 → N×N 矩阵可控 (≤ 5000²)
        A = self._build_adaptive_similarity(Z_spec)
        L = self._normalized_graph_laplacian(A)

        eigenvalues, eigenvectors = linalg.eigh(L)
        k_comp = min(self.n_states, eigenvectors.shape[1] - 1)
        spectral_embedding = eigenvectors[:, 1:k_comp+1]

        _, labels_spec, _ = k_means(spectral_embedding, n_clusters=self.n_states,
                                    init='k-means++', random_state=42, n_init=10)

        # 状态中心 (原始特征空间)
        self.state_centers = np.array([np.mean(Z_spec[labels_spec == c], axis=0)
                                       for c in range(self.n_states)])

        if n > max_spec:
            # 全量分配: 每个点到最近状态中心
            labels = np.zeros(n, dtype=int)
            min_dist = np.full(n, np.inf)
            for c in range(self.n_states):
                dist = np.sum((Z - self.state_centers[c])**2, axis=1)
                mask = dist < min_dist
                labels[mask] = c
                min_dist[mask] = dist[mask]
        else:
            labels = labels_spec

        self.state_labels = labels

        # 转移矩阵 (从全量标签估计)
        self.transition_matrix = np.zeros((self.n_states, self.n_states))
        for t in range(1, n):
            self.transition_matrix[labels[t-1], labels[t]] += 1
        row_sums = self.transition_matrix.sum(axis=1, keepdims=True)
        self.transition_matrix = np.where(row_sums > 0,
                                          self.transition_matrix / row_sums,
                                          np.eye(self.n_states)[labels[-1]])

        # 软概率
        proba = np.zeros((n, self.n_states))
        for i in range(n):
            for c in range(self.n_states):
                proba[i, c] = np.exp(-np.sum((Z[i] - self.state_centers[c])**2))
        proba = proba / (proba.sum(axis=1, keepdims=True) + 1e-10)

        return labels, proba

    def predict_one(self, z):
        """预测单个样本的状态概率"""
        if self.state_centers is None:
            return np.ones(self.n_states) / self.n_states
        proba = np.zeros(self.n_states)
        for c in range(self.n_states):
            proba[c] = np.exp(-np.sum((z - self.state_centers[c])**2))
        return proba / (proba.sum() + 1e-10)


class AKRRPredictor:
    """自适应核岭回归 — 状态条件预测 (Nyström 加速)

    每个隐状态下训练独立的 KRR 模型。
    当样本数 > nystrom_threshold 时使用 Nyström 低秩近似,
    将 N×N kernel 矩阵 (O(N²) 内存) 降为 N×m + m×m (O(Nm) 内存)。
    近似误差受 kernel 矩阵第 (m+1) 大特征值控制, m=2000 时通常 <0.1%。
    """
    def __init__(self, lambda_reg=1e-2, lambda_krr=1e-3, n_landmarks=2000):
        self.lambda_reg = lambda_reg
        self.lambda_krr = lambda_krr
        self.n_landmarks = n_landmarks
        self.models = {}

    def fit(self, Z, y, state_labels, n_states):
        for c in range(n_states):
            mask = state_labels == c
            n_c = np.sum(mask)
            if n_c < 5:
                continue
            Z_c = Z[mask]
            y_c = y[mask]

            # 马氏度量矩阵
            Sigma_c = np.cov(Z_c.T)
            Sigma_reg = Sigma_c + self.lambda_reg * np.eye(Sigma_c.shape[0])
            try:
                M_c = linalg.solve(Sigma_reg, np.eye(Sigma_c.shape[0]), assume_a='pos')
            except linalg.LinAlgError:
                M_c = linalg.pinv(Sigma_reg)

            gamma = 1.0 / (2 * Z_c.shape[1])

            if n_c <= 3000:
                # 小样本: 精确 KRR
                alpha_c = self._exact_krr_fit(Z_c, y_c, M_c, gamma)
                self.models[c] = {'alpha': alpha_c, 'Z_train': Z_c, 'M': M_c,
                                  'gamma': gamma, 'nystrom': False}
            else:
                # 大样本: Nyström 近似
                landmarks, beta = self._nystrom_krr_fit(Z_c, y_c, M_c, gamma)
                self.models[c] = {'landmarks': landmarks, 'beta': beta, 'M': M_c,
                                  'gamma': gamma, 'nystrom': True}

    # ── 精确 KRR ──
    def _exact_krr_fit(self, Z_c, y_c, M_c, gamma):
        XM = Z_c @ M_c
        XX = np.sum(XM * Z_c, axis=1)
        dist_sq = np.abs(XX[:, None] + XX[None, :] - 2 * (Z_c @ M_c @ Z_c.T))
        K_c = np.exp(-gamma * dist_sq)
        K_reg = K_c + self.lambda_krr * np.eye(K_c.shape[0])
        try:
            return linalg.solve(K_reg, y_c, assume_a='pos')
        except linalg.LinAlgError:
            return linalg.lstsq(K_reg, y_c)[0]

    # ── Nyström KRR ──
    def _nystrom_krr_fit(self, Z_c, y_c, M_c, gamma):
        """Nyström 加速 KRR 训练

        核心推导:
          K ≈ K_nm @ K_mm^{-1} @ K_nm^T = ΦΦ^T  (Φ = K_nm @ K_mm^{-1/2})
          (K + λI)^{-1}y = λ^{-1}[y - K_nm @ β]
          其中 β 满足: (λK_mm + K_nm^T K_nm) β = K_nm^T y

        预测时: y_pred = K_test_nm @ β  (只需 N_test × m 内存)
        """
        from sklearn.cluster import MiniBatchKMeans
        n, d = Z_c.shape
        m = min(self.n_landmarks, n // 2)

        # 选 landmark 点 (mini-batch k-means 质心)
        if n > 8000:
            rng = np.random.RandomState(42)
            sample_idx = rng.choice(n, min(n, 5000), replace=False)
            km = MiniBatchKMeans(n_clusters=m, random_state=42, batch_size=1024, n_init=1)
            km.fit(Z_c[sample_idx])
        else:
            km = MiniBatchKMeans(n_clusters=m, random_state=42, batch_size=1024, n_init=1)
            km.fit(Z_c)
        landmarks = km.cluster_centers_  # (m, d)

        # K_nm: kernel(所有训练点, landmarks)
        K_nm = self._kernel(Z_c, landmarks, M_c, gamma)  # (n, m)

        # K_mm: kernel(landmarks, landmarks)
        K_mm = self._kernel(landmarks, landmarks, M_c, gamma)  # (m, m)

        # 求解 β: (λK_mm + K_nm^T K_nm) β = K_nm^T y
        LHS = self.lambda_krr * K_mm + K_nm.T @ K_nm  # (m, m)
        RHS = K_nm.T @ y_c  # (m,)
        try:
            beta = linalg.solve(LHS, RHS, assume_a='pos')
        except linalg.LinAlgError:
            beta = linalg.lstsq(LHS, RHS)[0]

        # α (用于兼容旧接口, 不存储大矩阵)
        alpha_nystrom = (y_c - K_nm @ beta) / self.lambda_krr
        return landmarks, beta

    def _kernel(self, A, B, M_c, gamma):
        """马氏-RBF 核: k(a,b) = exp(-γ * d_M(a,b)²)"""
        XM = A @ M_c
        YM = B @ M_c
        XX = np.sum(XM * A, axis=1)
        YY = np.sum(YM * B, axis=1)
        dist_sq = np.abs(XX[:, None] + YY[None, :] - 2 * (A @ M_c @ B.T))
        return np.exp(-gamma * dist_sq)

    def predict(self, Z, state_proba):
        if Z.ndim == 1:
            Z = Z.reshape(1, -1)
        n = Z.shape[0]
        predictions = np.zeros(n)
        for c, model in self.models.items():
            M_c = model['M']
            gamma = model['gamma']
            if model.get('nystrom', False):
                # Nyström 预测: y_pred = K_test_nm @ β
                landmarks = model['landmarks']
                beta = model['beta']
                K_test_nm = self._kernel(Z, landmarks, M_c, gamma)  # (n, m)
                predictions += state_proba[:, c] * (K_test_nm @ beta)
            else:
                # 精确预测
                Z_train = model['Z_train']
                alpha = model['alpha']
                K_test = self._kernel(Z, Z_train, M_c, gamma)  # (n, N_train)
                predictions += state_proba[:, c] * (K_test @ alpha)
        return predictions


# ═══════════════════════════════════════════════
# 批量特征提取器 — 适配现有 pipeline
# ═══════════════════════════════════════════════

class SpectralMEnsemble:
    """SPECTRAL-M 集成模型: HOAT+VME → SSM → AKRR

    与现有 Stacking ensemble 接口兼容:
    - fit(X, y, returns_dict): 训练 SSM + AKRR
    - predict(X, returns_list): 预测
    - 可序列化 (pickle)
    """
    def __init__(self, hoat_window=60, vme_window=60, n_states=5,
                 akrr_lambda_reg=1e-2, akrr_lambda_krr=1e-3):
        self.hoat = HOATExtractor(window_size=hoat_window, n_components=8)
        self.vme = VMEmbedder(window_size=vme_window, local_window=20, n_components=6)
        self.ssm = SpectralStateMachine(n_states=n_states)
        self.akrr = AKRRPredictor(lambda_reg=akrr_lambda_reg, lambda_krr=akrr_lambda_krr)
        self._fitted = False
        self.feature_dim = 14

    def fit(self, returns_dict, targets_dict):
        """训练 SPECTRAL-M 模型

        Args:
            returns_dict: {stock_id: np.array of daily returns}
            targets_dict: {stock_id: np.array of target returns (aligned)}

        Returns:
            predictions: dict of {stock_id: np.array of predictions}
        """
        all_Z = []
        all_y = []
        all_stock_ids = []
        all_indices = []

        # 逐只股票提取 HOAT+VME 特征
        for sid in returns_dict:
            rets = returns_dict[sid]
            targets = targets_dict.get(sid)
            if targets is None or len(rets) < 90:
                continue

            try:
                hoat_feat, hoat_idx = self.hoat.fit_transform(rets)
                vme_feat, vme_idx = self.vme.fit_transform(rets)

                # 对齐 HOAT 和 VME 的时间索引
                common_idx = sorted(set(hoat_idx) & set(vme_idx))
                if len(common_idx) < 10:
                    continue

                hoat_aligned = hoat_feat[np.searchsorted(hoat_idx, common_idx)]
                vme_aligned = vme_feat[np.searchsorted(vme_idx, common_idx)]

                # 补齐到相同维度
                h_dim = min(hoat_aligned.shape[1], 8)
                v_dim = min(vme_aligned.shape[1], 6)
                Z_stock = np.zeros((len(common_idx), h_dim + v_dim), dtype=np.float32)
                Z_stock[:, :h_dim] = hoat_aligned[:, :h_dim]
                Z_stock[:, h_dim:h_dim+v_dim] = vme_aligned[:, :v_dim]

                # 对齐 targets
                tgt = targets[common_idx]

                all_Z.append(Z_stock)
                all_y.append(tgt)
                all_stock_ids.extend([sid] * len(common_idx))
                all_indices.extend(common_idx)

            except Exception:
                continue

        if not all_Z:
            return {}

        Z = np.vstack(all_Z).astype(np.float64)
        y = np.concatenate(all_y).astype(np.float64)

        # 标准化
        self.z_mean = Z.mean(axis=0)
        self.z_std = Z.std(axis=0) + 1e-8
        Z = (Z - self.z_mean) / self.z_std

        # SSM 拟合
        labels, proba = self.ssm.fit_predict(Z)

        # AKRR 训练
        self.akrr.fit(Z, y, labels, self.ssm.n_states)
        self._fitted = True

        # 生成预测
        all_preds = self.akrr.predict(Z, proba)

        # 按 stock_id 分组返回
        predictions = {}
        for i, sid in enumerate(all_stock_ids):
            if sid not in predictions:
                predictions[sid] = ([], [])
            predictions[sid][0].append(all_indices[i])
            predictions[sid][1].append(all_preds[i])

        result = {}
        for sid, (idx, preds) in predictions.items():
            result[sid] = (np.array(idx), np.array(preds))
        return result

    def predict_one(self, returns_series):
        """单只股票预测 — 用于推理"""
        if not self._fitted:
            return 0.0
        try:
            hoat_feat, hoat_idx = self.hoat.fit_transform(returns_series)
            vme_feat, vme_idx = self.vme.fit_transform(returns_series)
            if len(hoat_feat) == 0 or len(vme_feat) == 0:
                return 0.0

            # 取最后一个对齐的特征
            common_idx = sorted(set(hoat_idx) & set(vme_idx))
            if not common_idx:
                return 0.0
            last_idx = common_idx[-1]
            h_idx = hoat_idx.index(last_idx)
            v_idx = vme_idx.index(last_idx)

            h_dim = min(hoat_feat.shape[1], 8)
            v_dim = min(vme_feat.shape[1], 6)
            z = np.zeros(h_dim + v_dim, dtype=np.float64)
            z[:h_dim] = hoat_feat[h_idx, :h_dim]
            z[h_dim:h_dim+v_dim] = vme_feat[v_idx, :v_dim]

            z = (z - self.z_mean) / self.z_std
            proba = self.ssm.predict_one(z.reshape(1, -1)).flatten()
            return float(self.akrr.predict(z.reshape(1, -1), proba.reshape(1, -1))[0])
        except Exception:
            return 0.0
