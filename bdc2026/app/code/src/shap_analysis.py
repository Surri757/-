"""
SHAP 因子重要性分析
使用 TreeExplainer 对 LightGBM 模型进行 SHAP 分析，输出 top-20 特征。
"""
import os
import pickle
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def run_shap_analysis(lightgbm_model_path, X_background, feature_names, output_dir):
    """运行 SHAP 分析并输出结果

    Args:
        lightgbm_model_path: LightGBM 模型文件路径
        X_background: 背景数据 (n_samples, n_features)
        feature_names: 特征名列表
        output_dir: 输出目录

    Returns:
        top_features: [{'feature': name, 'shap_importance': float}, ...] top-20 列表
    """
    import shap

    # 加载模型
    with open(lightgbm_model_path, 'rb') as f:
        model = pickle.load(f)

    # Tree Explainer (path-dependent 最快)
    explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")

    # 计算 SHAP 值（采样加速）
    shap_values = explainer.shap_values(X_background)

    # 平均绝对 SHAP → 重要性
    if isinstance(shap_values, list):
        shap_values = shap_values[0]  # 回归通常是单输出

    importance = np.abs(shap_values).mean(axis=0)

    # Top-20 排序
    n_features = len(feature_names)
    n_top = min(20, n_features)
    top_indices = np.argsort(importance)[::-1][:n_top]

    top_features = []
    for idx in top_indices:
        top_features.append({
            'feature': str(feature_names[idx]) if idx < len(feature_names) else f'feature_{idx}',
            'shap_importance': float(importance[idx]),
        })

    # ── 输出 SHAP 重要性柱状图 ──
    fig, ax = plt.subplots(figsize=(10, 8))
    names = [f['feature'] for f in reversed(top_features)]
    values = [f['shap_importance'] for f in reversed(top_features)]
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(names)))
    ax.barh(range(len(names)), values, color=colors, edgecolor='white', height=0.7)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel('Mean |SHAP| Importance', fontsize=11)
    ax.set_title(f'Top {n_top} Feature Importances (SHAP)', fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    fig.tight_layout()

    shap_png = os.path.join(output_dir, 'shap_importance.png')
    fig.savefig(shap_png, dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ── 输出 JSON ──
    shap_json = os.path.join(output_dir, 'shap_importance.json')
    with open(shap_json, 'w', encoding='utf-8') as f:
        json.dump({'top_20': top_features}, f, indent=2, ensure_ascii=False)

    # ── 输出 summary plot (完整分布) ──
    try:
        fig2, _ = plt.subplots(figsize=(12, 8))
        shap.summary_plot(
            shap_values, X_background,
            feature_names=list(feature_names),
            max_display=20, show=False
        )
        summary_png = os.path.join(output_dir, 'shap_summary.png')
        fig2.savefig(summary_png, dpi=150, bbox_inches='tight')
        plt.close(fig2)
    except Exception:
        pass  # summary plot 失败不影响主流程

    top3_str = ', '.join(f['feature'] + '=' + str(round(f['shap_importance'], 4)) for f in top_features[:3])
    print(f"[SHAP] Top-3 features: {top3_str}")
    return top_features
