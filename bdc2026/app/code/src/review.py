"""
复盘层 — 记录、归因、报告
"""
import os
import json
from datetime import datetime
from collections import Counter
from typing import List, Dict


class ReviewLayer:
    """复盘记录 + 业绩归因

    记录:
    - Gate拒绝信号 (为什么被拒)
    - 执行成交记录
    - 风险事件 (止损/止盈/强制平仓)
    - 滑点偏差
    - 业绩归因
    """

    def __init__(self, output_dir: str = None):
        if output_dir is None:
            output_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'output'
            )
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.rejections: List[dict] = []
        self.executions: List[dict] = []
        self.risk_events: List[dict] = []
        self.slippage_data: List[dict] = []
        self.final_predictions: List[dict] = []

    def log_rejection(self, signal, reason: str):
        """记录Gate拒绝"""
        self.rejections.append({
            'date': signal.date,
            'stock_id': signal.stock_id,
            'reason': reason,
            'predicted_return': signal.predicted_return,
            'confidence': signal.confidence,
        })

    def log_execution(self, exec_record):
        """记录执行"""
        self.executions.append({
            'stock_id': exec_record.stock_id,
            'date': exec_record.date,
            'entry_price': exec_record.entry_price,
            'position_size': exec_record.position_size,
            'slippage': exec_record.slippage,
            'status': exec_record.status,
        })

    def log_risk_event(self, event: dict):
        """记录风险事件"""
        self.risk_events.append(event)

    def log_slippage(self, stock_id: str, expected: float, actual: float):
        """记录滑点"""
        self.slippage_data.append({
            'stock_id': stock_id,
            'expected_price': expected,
            'actual_price': actual,
            'slippage_pct': (actual - expected) / expected * 100 if expected > 0 else 0,
        })

    def log_prediction(self, stock_id: str, predicted_return: float, weight: float):
        """记录最终预测结果"""
        self.final_predictions.append({
            'stock_id': stock_id,
            'predicted_return': predicted_return,
            'weight': weight,
        })

    def compute_attribution(self) -> dict:
        """业绩归因"""
        if not self.final_predictions:
            return {'total_return': 0, 'n_stocks': 0}

        total_return = sum(
            p['weight'] * p['predicted_return']
            for p in self.final_predictions
        )
        n_stocks = len(self.final_predictions)
        win_count = sum(1 for p in self.final_predictions if p['predicted_return'] > 0)
        weights = [p['weight'] for p in self.final_predictions]
        rets = [p['predicted_return'] for p in self.final_predictions]

        return {
            'total_predicted_return': total_return,
            'n_stocks': n_stocks,
            'win_rate': win_count / max(n_stocks, 1),
            'max_weight': max(weights) if weights else 0,
            'min_weight': min(weights) if weights else 0,
            'total_exposure': sum(weights),
            'best_stock': max(self.final_predictions, key=lambda x: x['predicted_return'], default=None),
            'worst_stock': min(self.final_predictions, key=lambda x: x['predicted_return'], default=None),
        }

    def generate_report(self) -> str:
        """生成复盘报告"""
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        attr = self.compute_attribution()

        lines = [
            "=" * 60,
            "  量化交易复盘报告",
            f"  生成时间: {now}",
            "=" * 60,
            "",
            "--- 信号与Gate ---",
            f"  总信号数: {len(self.rejections) + len(self.executions)}",
            f"  通过Gate: {len(self.executions)}",
            f"  被拒绝:   {len(self.rejections)}",
        ]

        if self.rejections:
            rejection_counts = Counter(r['reason'].split(':')[0] for r in self.rejections)
            lines.append("  拒绝原因分布:")
            for reason, count in rejection_counts.most_common():
                lines.append(f"    - {reason}: {count}次")

        lines += [
            "",
            "--- 执行层 ---",
            f"  成交单数: {len(self.executions)}",
            f"  总仓位:   {attr['total_exposure']:.2%}",
        ]

        if self.slippage_data:
            avg_slip = sum(s['slippage_pct'] for s in self.slippage_data) / len(self.slippage_data)
            lines.append(f"  平均滑点: {avg_slip:.4f}%")

        lines += [
            "",
            "--- 风控层 ---",
            f"  风险事件: {len(self.risk_events)}",
        ]
        if self.risk_events:
            event_types = Counter(e.get('type', e.get('event_type', 'unknown')) for e in self.risk_events)
            for etype, count in event_types.most_common():
                lines.append(f"    - {etype}: {count}次")

        lines += [
            "",
            "--- 最终组合 ---",
            f"  持仓数:     {attr['n_stocks']}",
            f"  正向率:     {attr['win_rate']:.1%}",
            f"  总仓位:     {attr['total_exposure']:.2%}",
            f"  总预测收益率: {attr['total_predicted_return']:.4%}",
        ]

        # 股票明细
        if self.final_predictions:
            lines.append("")
            lines.append(f"  {'股票':<8} {'权重':>8} {'预测收益':>10}")
            lines.append("  " + "-" * 28)
            for p in sorted(self.final_predictions, key=lambda x: x['weight'], reverse=True):
                lines.append(f"  {p['stock_id']:<8} {p['weight']:>7.4%} {p['predicted_return']:>+9.4%}")

        lines += [
            "",
            "=" * 60,
            "  复盘完成",
            "=" * 60,
        ]

        report = "\n".join(lines)
        print(report)

        # 保存到文件
        report_path = os.path.join(
            self.output_dir,
            f"review_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)

        # 同时保存JSON格式供后续分析
        json_path = os.path.join(
            self.output_dir,
            f"review_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({
                'attribution': attr,
                'rejections': len(self.rejections),
                'executions': len(self.executions),
                'risk_events': len(self.risk_events),
                'predictions': self.final_predictions,
            }, f, ensure_ascii=False, indent=2, default=str)

        return report
