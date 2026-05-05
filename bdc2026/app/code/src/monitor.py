"""
训练可视化监控 — 损失曲线 / IC曲线 / 特征重要性 / 模型对比仪表盘
"""
import os
import numpy as np
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# 中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class TrainingMonitor:
    """训练过程监控器"""

    def __init__(self, output_dir=None):
        if output_dir is None:
            output_dir = os.path.dirname(os.path.abspath(__file__))
            output_dir = os.path.join(os.path.dirname(output_dir), 'output')
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.pt_history = defaultdict(list)   # PyTorch训练历史
        self.cv_history = []                   # CV折历史
        self.base_metrics = {}                 # 各模型最终指标
        self.feature_importance = None
        self.meta_weights = None

    def log_pt_epoch(self, model_name, epoch, train_loss, val_loss, val_ic):
        """记录PyTorch每个epoch的指标"""
        self.pt_history[model_name].append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_ic': val_ic,
        })

    def log_cv_fold(self, fold, train_size, val_size, model_ics):
        """记录CV折信息"""
        self.cv_history.append({
            'fold': fold,
            'train_size': train_size,
            'val_size': val_size,
            'model_ics': model_ics,
        })

    def log_base_metrics(self, model_name, metrics):
        """记录基础模型最终指标"""
        self.base_metrics[model_name] = metrics

    def log_feature_importance(self, feature_names, importance):
        """记录特征重要性"""
        self.feature_importance = (feature_names, importance)

    def log_meta_weights(self, model_names, weights):
        """记录元模型权重"""
        self.meta_weights = (model_names, weights)

    def plot_pt_curves(self):
        """绘制PyTorch模型训练曲线"""
        if not self.pt_history:
            return

        for name, history in self.pt_history.items():
            if len(history) < 2:
                continue

            epochs = [h['epoch'] for h in history]
            train_loss = [h['train_loss'] for h in history]
            val_loss = [h['val_loss'] for h in history]
            val_ic = [h['val_ic'] for h in history]

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

            # Loss子图
            ax1.plot(epochs, train_loss, 'b-', alpha=0.6, linewidth=1.5, label='Train Loss')
            ax1.plot(epochs, val_loss, 'r-', linewidth=2, label='Val Loss')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.set_title(f'{name.upper()} — Loss')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

            # IC子图
            ax2.plot(epochs, val_ic, 'g-', linewidth=2, label='Val IC')
            ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('IC')
            ax2.set_title(f'{name.upper()} — Information Coefficient')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.xaxis.set_major_locator(MaxNLocator(integer=True))

            plt.tight_layout()
            path = os.path.join(self.output_dir, f'training_curve_{name}.png')
            fig.savefig(path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"  训练曲线已保存: {path}")

    def plot_model_comparison(self):
        """模型对比柱状图"""
        if not self.base_metrics:
            return

        names = list(self.base_metrics.keys())
        ics = [self.base_metrics[n].get('IC', 0) for n in names]
        mses = [self.base_metrics[n].get('MSE', 0) for n in names]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(names)))

        # IC对比
        bars1 = ax1.bar(range(len(names)), ics, color=colors, edgecolor='white', linewidth=0.8)
        ax1.set_xticks(range(len(names)))
        ax1.set_xticklabels(names, rotation=30, ha='right', fontsize=9)
        ax1.set_ylabel('IC')
        ax1.set_title('模型 IC 对比')
        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        for bar, val in zip(bars1, ics):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                     f'{val:.4f}', ha='center', va='bottom', fontsize=8)

        # MSE对比
        bars2 = ax2.bar(range(len(names)), mses, color=colors, edgecolor='white', linewidth=0.8)
        ax2.set_xticks(range(len(names)))
        ax2.set_xticklabels(names, rotation=30, ha='right', fontsize=9)
        ax2.set_ylabel('MSE')
        ax2.set_title('模型 MSE 对比')
        for bar, val in zip(bars2, mses):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.00002,
                     f'{val:.6f}', ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        path = os.path.join(self.output_dir, 'model_comparison.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  模型对比图已保存: {path}")

    def plot_meta_weights(self):
        """元模型权重饼图"""
        if self.meta_weights is None:
            return

        names, weights = self.meta_weights
        # 过滤掉压缩的
        active = [(n, w) for n, w in zip(names, weights) if w > 0.01]
        if not active:
            return

        names, weights = zip(*active)
        colors = plt.cm.Set3(np.linspace(0, 1, len(names)))

        fig, ax = plt.subplots(figsize=(8, 8))
        wedges, texts, autotexts = ax.pie(
            weights, labels=names, autopct='%1.1f%%',
            colors=colors, startangle=90,
            textprops={'fontsize': 10}
        )
        for at in autotexts:
            at.set_fontweight('bold')
            at.set_fontsize(9)
        ax.set_title('Stacking 元模型权重分配', fontsize=13, fontweight='bold')

        path = os.path.join(self.output_dir, 'meta_weights.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  权重饼图已保存: {path}")

    def plot_feature_importance(self, top_n=30):
        """特征重要性Top-N图"""
        if self.feature_importance is None:
            return

        feat_names, importance = self.feature_importance
        if len(feat_names) == 0:
            return

        # 取Top N (不超过实际特征数)
        top_n = min(top_n, len(feat_names))
        idx = np.argsort(importance)[-top_n:]
        top_feat = [feat_names[i] for i in idx]
        top_imp = [importance[i] for i in idx]

        fig, ax = plt.subplots(figsize=(10, max(8, top_n * 0.28)))
        colors = plt.cm.viridis(np.linspace(0.2, 0.9, top_n))
        ax.barh(range(top_n), top_imp, color=colors, edgecolor='white', linewidth=0.5)
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(top_feat, fontsize=8)
        ax.set_xlabel('Importance')
        ax.set_title(f'特征重要性 Top {top_n}', fontsize=12, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)

        plt.tight_layout()
        path = os.path.join(self.output_dir, 'feature_importance.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  特征重要性图已保存: {path}")

    def generate_dashboard(self):
        """生成完整监控仪表盘"""
        print("\n[监控] 生成训练可视化图表...")
        self.plot_pt_curves()
        self.plot_model_comparison()
        self.plot_meta_weights()
        self.plot_feature_importance()
        print("[监控] 可视化图表生成完成")
