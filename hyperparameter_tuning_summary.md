# 超参数调优策略汇总

## 1. 超参数调优方法对比

### 1.1 Grid Search (网格搜索)
- **原理**: 遍历所有参数组合
- **优点**: 保证找到全局最优
- **缺点**: 计算成本高，维度灾难
- **适用场景**: 参数空间较小时

### 1.2 Random Search (随机搜索)
- **原理**: 随机采样参数组合
- **优点**: 比网格搜索更高效，尤其是高维空间
- **缺点**: 可能错过最优解
- **适用场景**: 参数空间较大时

### 1.3 Bayesian Optimization (贝叶斯优化)
- **原理**: 使用代理模型(surrogate model)学习参数与性能的关系
- **优点**: 
  - 高效利用评估次数
  - 处理高维参数空间
  - 能处理非凸、不连续的目标函数
- **缺点**: 需要足够的初始探索
- **适用场景**: 评估成本高的情况

## 2. XGBoost 关键超参数调优指南

### 2.1 核心参数

| 参数 | 说明 | 推荐范围 | 调优建议 |
|------|------|----------|----------|
| `max_depth` | 树的最大深度 | 3-10 | 防止过拟合，通常3-6 |
| `learning_rate` (eta) | 学习率 | 0.01-0.3 | 较低值需要更多树 |
| `n_estimators` | 树的数量 | 100-1000 | 与learning_rate配合 |
| `min_child_weight` | 最小叶子节点权重 | 1-10 | 防止过拟合 |
| `subsample` | 行采样比例 | 0.5-1.0 | 通常0.8 |
| `colsample_bytree` | 列采样比例 | 0.5-1.0 | 通常0.8 |
| `gamma` | 最小损失减少 | 0-5 | 正则化参数 |
| `reg_alpha` (L1) | L1正则化 | 0-1 | 稀疏化特征 |
| `reg_lambda` (L2) | L2正则化 | 1-∞ | 权重平滑 |

### 2.2 调优策略（推荐顺序）

**Step 1: 固定学习率，调优树参数**
```python
# 固定learning_rate=0.1，使用early_stopping找最佳n_estimators
xgb_model = XGBoost(learning_rate=0.1, n_estimators=1000, early_stopping_rounds=50)
```

**Step 2: 调优max_depth和min_child_weight**
- max_depth: 3-10
- min_child_weight: 1-10

**Step 3: 调优subsample和colsample_bytree**
- subsample: 0.6-1.0
- colsample_bytree: 0.6-1.0

**Step 4: 调优正则化参数**
- gamma: 0-5
- reg_alpha: 0-1
- reg_lambda: 1-10

**Step 5: 降低学习率，增加树数量**
- 最终微调

## 3. LightGBM 关键超参数调优指南

### 3.1 核心参数

| 参数 | 说明 | 推荐范围 |
|------|------|----------|
| `num_leaves` | 叶子节点数 | 20-100 |
| `max_depth` | 树的最大深度 | 5-15 |
| `learning_rate` | 学习率 | 0.01-0.2 |
| `n_estimators` | 迭代次数 | 100-1000 |
| `min_child_samples` | 叶子最小数据量 | 10-50 |
| `subsample` | 行采样比例 | 0.5-1.0 |
| `colsample_bytree` | 列采样比例 | 0.5-1.0 |
| `reg_alpha` | L1正则化 | 0-1 |
| `reg_lambda` | L2正则化 | 0-10 |

### 3.2 Leaf-wise策略
LightGBM使用leaf-wise（最佳优先）生长策略，比level-wise更高效：
```python
# 推荐配置
lgb_params = {
    'num_leaves': 31,  # 不要超过2^(max_depth+1)
    'max_depth': -1,   # 不限制
    'learning_rate': 0.05,
    'n_estimators': 500,
    'min_child_samples': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.8
}
```

## 4. Optuna 自动超参数优化框架

### 4.1 特点
- **动态搜索空间**: 根据试验结果动态调整搜索范围
- **多种采样算法**: TPE (Tree-structured Parzen Estimator)、CMA-ES、随机搜索
- **剪枝策略**: 自动停止表现不佳的试验
- **支持多种ML框架**: XGBoost、LightGBM、PyTorch等

### 4.2 XGBoost + Optuna示例
```python
import optuna
import xgboost as xgb
from sklearn.model_selection import cross_val_score

def objective(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
    }
    
    model = xgb.XGBClassifier(**params)
    scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
    return scores.mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
```

## 5. 金融量化模型特有考虑

### 5.1 时间序列交叉验证
- 使用Walk-Forward Validation而非简单的K-Fold
- 避免未来数据泄露
```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
for train_idx, val_idx in tscv.split(X):
    # 训练和验证
```

### 5.2 金融数据调参注意事项
1. **金融数据非平稳性**: 考虑滚动窗口重训
2. **过拟合风险**: 金融数据噪声大，需要更强的正则化
3. **交易成本**: 评估指标需考虑手续费、滑点
4. **样本不平衡**: 涨跌幅分布通常不对称

### 5.3 推荐评估指标
- Sharpe Ratio (夏普比率)
- Max Drawdown (最大回撤)
- Sortino Ratio
- Calmar Ratio
- IC (Information Coefficient)

## 6. 贝叶斯优化实战

### 6.1 使用hyperopt库
```python
from hyperopt import fmin, tpe, hp, Trials

space = {
    'max_depth': hp.choice('max_depth', range(3, 10)),
    'learning_rate': hp.loguniform('learning_rate', -3, 0),
    'n_estimators': hp.choice('n_estimators', [100, 200, 300, 500]),
    'min_child_weight': hp.choice('min_child_weight', range(1, 10)),
    'subsample': hp.uniform('subsample', 0.5, 1.0),
}

def objective(params):
    model = xgb.XGBClassifier(**params)
    score = cross_val_score(model, X, y, cv=3, scoring='roc_auc').mean()
    return -score

trials = Trials()
best = fmin(fn=objective, space=space, algo=tpe.suggest, trials=trials, max_evals=100)
```

### 6.2 使用scikit-optimize
```python
from skopt import BayesSearchCV
from skopt.space import Real, Integer

search_space = {
    'max_depth': Integer(3, 10),
    'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
    'n_estimators': Integer(100, 500),
}

opt = BayesSearchCV(
    xgb.XGBClassifier(),
    search_space,
    n_iter=50,
    cv=3,
    scoring='roc_auc'
)
opt.fit(X, y)
```

## 7. 参考资料

1. LightGBM官方参数调优文档: https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html
2. XGBoost超参数调优指南 (Analytics Vidhya): https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/
3. XGBoost贝叶斯优化调参: https://aiinpractice.com/xgboost-hyperparameter-tuning-with-bayesian-optimization/
4. Optuna官方文档: https://optuna.org/
5. Hyperopt GitHub: https://github.com/hyperopt/hyperopt
6. 中国股票趋势预测贝叶斯优化研究: https://www.atlantis-press.com/proceedings/icdeba-24/126008524

## 8. 实用建议总结

1. **从简单开始**: 先用默认参数建立baseline
2. **优先调learning_rate和n_estimators**: 这两个参数影响最大
3. **使用early_stopping**: 避免过拟合
4. **考虑时间成本**: 贝叶斯优化适合评估成本高的场景
5. **金融数据特殊处理**: 使用时间序列交叉验证
6. **多指标评估**: 不仅看AUC，也要看实际交易表现
