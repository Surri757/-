"""
多层筛选流水线 — 组内竞争 + 跨组归一 + 复活赛 + 决赛优化

Stage 1 (粗筛): 300 → ~150, 组内不同模型打分
Stage 2 (精选): ~150 → ~45, 换一套模型重新打分
复活赛:         淘汰池 → 最多 5 只, TFT+TimesNet 救回
Stage 3 (决赛): ~50 → 5, 全量 7 模型 Stacking + 组合优化

核心理念: 不同阶段的模型组合不同，避免单一模型偏见。
组内用 z-score 排名，跨组可比。
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
from tqdm import tqdm


# ═══════════════════════════════════════════════
# 硬黑名单: 压舱石/低活跃度股票，第一轮直接淘汰
# ═══════════════════════════════════════════════

BALLAST_BLACKLIST = {
    # ── 国有银行 + 股份制银行 (低活跃压舱石, 41只) ──
    '601398', '601939', '601288', '601988',  # 工商/建设/农业/中国
    '600036', '600000', '600016', '600015',  # 招商/浦发/民生/华夏
    '601009', '601166', '601818', '601229',  # 南京/兴业/光大/上海
    '600919', '002142', '000001',            # 江苏/宁波/平安
    '601998', '601997', '601169', '600926',  # 中信/贵阳/北京/杭州
    '002839', '600908', '002948',            # 张家港/无锡/青岛
    '601328', '601658', '601838', '601077',  # 交通/邮储/成都/渝农
    '601528', '600928', '601825', '601860',  # 瑞丰/西安/上海农商/紫金
    '601963', '002936', '600015', '601916',  # 重庆/郑州/华夏/浙商
    '601187', '600036', '601995',            # 厦门/招商(重)/中金
    # ── 石油石化+能源 (传统权重) ──
    '601857', '600028', '600938', '601088',  # 中国石油/中国石化/中国海油/中国神华
    '601808', '600968',                       # 中海油服/海油发展
    # ── 运营商 (类债券走势) ──
    '600941', '600050', '601728',            # 中国移动/联通/电信
    # ── 电力公用事业 (常年横盘稳分红) ──
    '600900', '601985', '600886', '600674',  # 长江电力/中国核电/国投电力/川投能源
    '600025', '600023', '600011', '600027',  # 华能水电/浙能电力/华能国际/华电国际
    '601991', '600795', '600886',            # 大唐发电/国电电力
    '003816', '000883', '000027',            # 中国广核/湖北能源/深圳能源
    # ── 老牌保险 ──
    '601318', '601628', '601336', '601601',  # 中国平安/中国人寿/新华保险/中国太保
    '601319',                                # 中国人保
    # ── 中字头基建+建筑 ──
    '601668', '601390', '601186', '601800',  # 中国建筑/中国中铁/中国铁建/中国交建
    '601618', '601669', '600170',            # 中国中冶/中国电建/上海建工
    '601868', '601117',                       # 中国能建/中国化学
    # ── 铁路+高速+港口 (类债券, 低活跃) ──
    '601006', '600377', '001965', '600350',  # 大秦铁路/宁沪高速/招商公路/山东高速
    '600012', '600548', '600009',            # 皖通高速/深高速/上海机场
    '601018', '601298', '601880',            # 宁波港/青岛港/大连港(辽港)
    '600018',                                # 上港集团
    # ── 其他大市值低活跃国企 ──
    '600104', '601238', '601633',            # 上汽集团/广汽集团/长城汽车
    '600519', '000858',                       # 贵州茅台/五粮液 (超大盘消费, 无爆发力)
}


# ═══════════════════════════════════════════════
# 因子画像分组器
# ═══════════════════════════════════════════════

class StockGrouper:
    """基于因子画像将股票分组：波动率 × 动量 × 估值"""

    # 分组标签映射
    GROUP_LABELS = {
        (0, 1, 1): 'lowvol_growth',     # 低波+成长 → 慢牛
        (0, 0, 1): 'lowvol_value_growth', # 低波+价值成长混合
        (0, 0, 0): 'lowvol_value',       # 低波+价值 → 防御
        (1, 1, 1): 'midvol_growth',      # 中波+成长 → 稳步上行
        (1, 1, 0): 'midvol_mixed',       # 中波+混合
        (1, 0, 0): 'midvol_value',       # 中波+价值 → 均值回归
        (2, 1, 1): 'highvol_momentum',   # 高波+强动量 → 趋势爆发
        (2, 0, 0): 'highvol_reversal',   # 高波+弱动量 → 超跌反弹
    }

    def __init__(self, df, signals):
        """
        Args:
            df: 完整 stock DataFrame (含 close, pe, pb)
            signals: 已生成的 StockSignal 列表
        """
        self.df = df
        self.signals = signals
        self.group_map = {}      # stock_id → group_name
        self.group_members = {}  # group_name → [stock_ids]

    def assign_groups(self):
        """为所有信号中的股票分配分组"""
        if not self.signals:
            return

        stock_ids = list(set(s.stock_id for s in self.signals))

        # 计算每只股票的因子值
        momentum_map = self._compute_momentum(stock_ids)
        value_map = self._compute_value_type(stock_ids)

        for sid in stock_ids:
            s = next((x for x in self.signals if x.stock_id == sid), None)
            if s is None:
                continue
            vol_cluster = s.volatility_cluster  # 0低/1中/2高
            momentum = momentum_map.get(sid, 1)  # 0弱/1强
            value = value_map.get(sid, 1)        # 0价值/1成长

            # 找到最匹配的分组标签
            group = self._match_group(vol_cluster, momentum, value)
            self.group_map[sid] = group

        # 构建分组成员表
        self.group_members = defaultdict(list)
        for sid, g in self.group_map.items():
            self.group_members[g].append(sid)

        # 合并小分组到相邻组（单只股票的组没有比较意义）
        self._merge_small_groups()

    def _match_group(self, vol, momentum, value):
        """模糊匹配分组: 精确匹配失败则放宽"""
        # 先精确匹配
        key = (vol, momentum, value)
        if key in self.GROUP_LABELS:
            return self.GROUP_LABELS[key]
        # 放宽估值维度
        for v in [value, 1 - value]:
            key2 = (vol, momentum, v)
            if key2 in self.GROUP_LABELS:
                return self.GROUP_LABELS[key2]
        # 再放宽动量维度
        for m in [momentum, 1 - momentum]:
            for v in [value, 1 - value]:
                key3 = (vol, m, v)
                if key3 in self.GROUP_LABELS:
                    return self.GROUP_LABELS[key3]
        return 'midvol_mixed'  # 兜底

    def _merge_small_groups(self):
        """将成员少于 3 只的小组合并到最相似的大组"""
        small = [g for g, members in self.group_members.items() if len(members) < 3]
        for sg in small:
            members = self.group_members.pop(sg, [])
            # 找到最相似的大组
            best_group = max(self.group_members.items(),
                           key=lambda x: self._group_similarity(sg, x[0]),
                           default=(None, []))[0]
            if best_group:
                self.group_members[best_group].extend(members)
                for sid in members:
                    self.group_map[sid] = best_group

    @staticmethod
    def _group_similarity(g1, g2):
        """两个分组的相似度（基于标签关键词）"""
        parts1, parts2 = set(g1.split('_')), set(g2.split('_'))
        return len(parts1 & parts2)

    def _compute_momentum(self, stock_ids):
        """计算动量分位: 0弱(后50%) / 1强(前50%)"""
        momentum = {}
        returns_list = []
        sid_list = []
        for sid in stock_ids:
            sub = self.df[self.df['stock_id'].astype(str).str.zfill(6) == sid]
            if len(sub) < 60:
                continue
            sub = sub.sort_values('date')
            close = sub['close'].values
            ret_60d = (close[-1] / close[-60] - 1) if len(close) >= 60 else 0
            returns_list.append(ret_60d)
            sid_list.append(sid)

        if not returns_list:
            return {sid: 1 for sid in stock_ids}

        median = np.median(returns_list)
        for sid, ret in zip(sid_list, returns_list):
            momentum[sid] = 1 if ret > median else 0
        # 未覆盖的股票默认中性
        for sid in stock_ids:
            if sid not in momentum:
                momentum[sid] = 1
        return momentum

    def _compute_value_type(self, stock_ids):
        """估值类型: 0价值(低PE) / 1成长(高PE)"""
        value = {}
        pe_list = []
        sid_list = []
        for sid in stock_ids:
            sub = self.df[self.df['stock_id'].astype(str).str.zfill(6) == sid]
            if len(sub) < 1:
                continue
            sub = sub.sort_values('date')
            pe = sub['pe'].values[-1] if 'pe' in sub.columns and pd.notna(sub['pe'].values[-1]) else None
            if pe is not None and pe > 0:
                pe_list.append(pe)
                sid_list.append(sid)

        if not pe_list:
            return {sid: 1 for sid in stock_ids}

        median = np.median(pe_list)
        for sid, pe in zip(sid_list, pe_list):
            value[sid] = 1 if pe > median else 0  # 1=成长(高PE), 0=价值(低PE)
        for sid in stock_ids:
            if sid not in value:
                value[sid] = 1
        return value

    def get_group(self, stock_id):
        return self.group_map.get(stock_id, 'midvol_mixed')

    def summary(self):
        lines = [f"[分组] {len(self.group_members)} 组:"]
        for g, members in sorted(self.group_members.items(), key=lambda x: -len(x[1])):
            lines.append(f"  {g}: {len(members)} 只")
        return '\n'.join(lines)


# ═══════════════════════════════════════════════
# 阶段模型配置
# ═══════════════════════════════════════════════

class StageConfig:
    """每个阶段、每个分组使用哪些模型"""

    # Stage 1: 粗筛 — 各组用最适配的 2 个模型
    STAGE1_MODELS = {
        'highvol_momentum':      ['tft', 'patchtst'],
        'highvol_reversal':      ['tft', 'xgboost'],
        'midvol_growth':         ['lightgbm', 'dlinear'],
        'midvol_mixed':          ['catboost', 'lightgbm'],
        'midvol_value':          ['catboost', 'lightgbm'],
        'lowvol_growth':         ['tft', 'catboost'],
        'lowvol_value':          ['lightgbm', 'xgboost'],
        'lowvol_value_growth':   ['dlinear', 'catboost'],
    }

    # Stage 2: 精选 — 换不同模型族交叉验证
    STAGE2_MODELS = {
        'highvol_momentum':      ['catboost', 'dlinear'],
        'highvol_reversal':      ['lightgbm', 'tft'],
        'midvol_growth':         ['tft', 'xgboost'],
        'midvol_mixed':          ['dlinear', 'xgboost'],
        'midvol_value':          ['xgboost', 'dlinear'],
        'lowvol_growth':         ['dlinear', 'lightgbm'],
        'lowvol_value':          ['catboost', 'dlinear'],
        'lowvol_value_growth':   ['xgboost', 'tft'],
    }

    # 复活赛: DL 模型专救被树模型误杀的
    RESURRECTION_MODELS = ['tft', 'dlinear']

    @classmethod
    def get_models(cls, stage, group):
        config = cls.STAGE1_MODELS if stage == 1 else cls.STAGE2_MODELS
        return config.get(group, ['lightgbm', 'catboost'])

    @classmethod
    def get_all_used_models(cls, stage):
        """获取某阶段所有用到的模型名"""
        config = cls.STAGE1_MODELS if stage == 1 else cls.STAGE2_MODELS
        models = set()
        for names in config.values():
            models.update(names)
        return models


# ═══════════════════════════════════════════════
# 复活赛评审
# ═══════════════════════════════════════════════

class ResurrectionJudge:
    """复活赛: 用 DL 模型重新审视被淘汰的股票"""

    def __init__(self, predictor, feature_eng, seq_len=60):
        self.predictor = predictor        # StackingPredictor，可调用单个模型
        self.fe = feature_eng
        self.seq_len = seq_len

    def evaluate(self, eliminated_signals, stock_df, max_rescue=5):
        """对淘汰池中的股票重新打分，选出复活者

        Args:
            eliminated_signals: 被淘汰的 StockSignal 列表
            stock_df: 完整股票数据
            max_rescue: 最多复活数量

        Returns:
            rescued_signals: 复活成功的信号列表
        """
        if len(eliminated_signals) <= max_rescue:
            return eliminated_signals  # 淘汰池太小，全复活

        resurrection_models = StageConfig.RESURRECTION_MODELS
        scores = []

        for signal in tqdm(eliminated_signals, desc="  复活评审", unit="stock", leave=False):
            # 为每只股票单独打分
            stock_data = stock_df[
                stock_df['stock_id'].astype(str).str.zfill(6) == signal.stock_id
            ].sort_values('date').set_index('date')

            if len(stock_data) < self.seq_len:
                scores.append((signal, 0.0))
                continue

            # 特征工程
            features = self.fe.build_all_features(stock_data, None, None)
            features = self.fe.remove_outliers(features)
            last_features = features.iloc[-self.seq_len:].fillna(0)
            X = last_features.values.astype(np.float32)

            # 用复活赛 DL 模型打分
            model_preds = []
            for model_name in resurrection_models:
                try:
                    pred = self._predict_with_model(model_name, X)
                    model_preds.append(pred)
                except Exception:
                    continue

            if not model_preds:
                scores.append((signal, 0.0))
                continue

            # 复活分 = 模型预测均值
            rescue_score = float(np.mean(model_preds))
            # 加分项: 如果 DL 预测远高于信号原始 prediction（树模型低估）
            boost = max(0, rescue_score - signal.predicted_return)
            rescue_score += boost * 0.5

            scores.append((signal, rescue_score))

        # 按复活分降序，取 top max_rescue
        scores.sort(key=lambda x: x[1], reverse=True)
        rescued = [s for s, sc in scores[:max_rescue] if sc > 0.0]

        if rescued:
            names = ', '.join(f'{s.stock_id}({sc:.4f})' for s, sc in scores[:max_rescue])
            print(f"[复活赛] 救回 {len(rescued)} 只: {names}")

        return rescued

    def _predict_with_model(self, model_name, X):
        """用指定模型预测"""
        model = self.predictor.base_models.get(model_name)
        if model is None:
            return 0.0

        X_scaled = self.predictor.scaler.transform(X) if self.predictor.scaler else X
        X_t = np.expand_dims(X_scaled, axis=0).astype(np.float32)

        if model_name in ['lightgbm', 'catboost', 'xgboost', 'spectralm']:
            # GBDT: 展平输入
            return float(model.predict(X_scaled[-1:])[0])
        else:
            # PyTorch: 时序输入
            import torch
            with torch.no_grad():
                t = torch.from_numpy(X_t).to(self.predictor.device)
                return float(model(t).cpu().numpy())


# ═══════════════════════════════════════════════
# 动态活跃度过滤器
# ═══════════════════════════════════════════════

class DynamicActivityFilter:
    """动态低活跃度过滤: 30日内交易不活跃的票暂时拉黑, 放量则复活

    与永久黑名单不同: 永久黑名单是结构性的(压舱石), 动态黑名单是阶段性的。
    如果股票近期出现持续放量 → 自动解除动态黑名单, 重新进入候选池。
    """

    def __init__(self, stock_df):
        self.stock_df = stock_df
        self.dynamic_blacklist = set()
        self.resurrected = set()  # 被复活(放量)的股票

    def evaluate(self, signals, lookback_short=30, lookback_long=120):
        """评估每只股票的活跃度, 低活跃的拉入动态黑名单

        Args:
            signals: 当前信号列表
            lookback_short: 短期窗口(交易日) → 默认30天
            lookback_long: 长期窗口(交易日) → 默认120天

        Returns:
            active_signals: 通过活跃度检查的信号
            low_activity_signals: 被动态拉黑的信号
        """
        active_signals = []
        low_activity_signals = []
        self.dynamic_blacklist = set()
        self.resurrected = set()

        for signal in signals:
            sid = signal.stock_id
            sub = self.stock_df[
                self.stock_df['stock_id'].astype(str).str.zfill(6) == sid
            ].sort_values('date')

            if len(sub) < lookback_short:
                active_signals.append(signal)  # 数据不足, 放行
                continue

            # 成交量/换手率
            if 'volume' in sub.columns:
                vol_data = sub['volume'].values
            elif 'turnover_rate' in sub.columns:
                vol_data = sub['turnover_rate'].values
            else:
                vol_data = sub['amount'].values

            # 长期 vs 短期均量对比
            long_avg = np.mean(vol_data[-lookback_long:]) if len(vol_data) >= lookback_long else np.mean(vol_data)
            short_avg = np.mean(vol_data[-lookback_short:])

            # 检查最近一周是否有放量: 任一天成交量 > 长期中位数的2倍
            recent = vol_data[-5:]  # 最近5个交易日
            long_median = np.median(vol_data[-lookback_long:]) if len(vol_data) >= lookback_long else np.median(vol_data)
            has_surge = np.any(recent > long_median * 2.0)
            sustained_surge = np.sum(recent > long_median * 1.5) >= 2  # 2天以上持续放量

            if short_avg < long_avg * 0.6:
                # 近期活跃度显著低于自身历史 → 动态拉黑
                if has_surge or sustained_surge:
                    # 但有放量信号 → 复活!
                    self.resurrected.add(sid)
                    active_signals.append(signal)
                else:
                    self.dynamic_blacklist.add(sid)
                    low_activity_signals.append(signal)
            else:
                active_signals.append(signal)

        return active_signals, low_activity_signals

    def summary(self):
        return (f"[动态活跃度] 拉黑 {len(self.dynamic_blacklist)} 只(近期不活跃), "
                f"复活 {len(self.resurrected)} 只(放量突破)")


# ═══════════════════════════════════════════════
# 多层筛选主管道
# ═══════════════════════════════════════════════

@dataclass
class StageResult:
    """每阶段的筛选结果"""
    stage: int
    survivors: List        # 存活的 StockSignal
    eliminated: List       # 被淘汰的 StockSignal
    group_scores: Dict[str, List[Tuple]]  # group → [(signal, z_score)]


class CascadePipeline:
    """多层筛选主控"""

    def __init__(self, predictor, feature_eng, industry_df, macro_df, seq_len=60):
        self.predictor = predictor
        self.fe = feature_eng
        self.industry_df = industry_df
        self.macro_df = macro_df
        self.seq_len = seq_len

        self.grouper = None
        self.resurrection = ResurrectionJudge(predictor, feature_eng, seq_len)
        self.results: List[StageResult] = []

    def run(self, signals, stock_df, verbose=True):
        """执行完整的多层筛选流水线

        Args:
            signals: 所有生成的 StockSignal 列表
            stock_df: 完整股票 DataFrame

        Returns:
            final_candidates: 决赛候选信号列表（≤55 只）
            all_results: 各阶段结果
        """
        if len(signals) < 10:
            return signals, []

        # ── 永久黑名单: 压舱石直接淘汰，不可复活 ──
        non_ballast = []
        n_blacklisted = 0
        for s in signals:
            if s.stock_id in BALLAST_BLACKLIST:
                n_blacklisted += 1
            else:
                non_ballast.append(s)
        if verbose and n_blacklisted > 0:
            blacklisted_names = [s.stock_id for s in signals if s.stock_id in BALLAST_BLACKLIST]
            print(f"[永久黑名单] 淘汰 {n_blacklisted} 只压舱石: {', '.join(blacklisted_names[:8])}"
                  + (f' ...等{len(blacklisted_names)}只' if len(blacklisted_names) > 8 else ''))
        signals = non_ballast

        if len(signals) < 10:
            return signals, []

        # ── 动态活跃度过滤: 近期不活跃暂时拉黑, 放量自动复活 ──
        activity_filter = DynamicActivityFilter(stock_df)
        signals, low_act = activity_filter.evaluate(signals)
        if verbose:
            print(activity_filter.summary())
            if len(low_act) > 0:
                names = [s.stock_id for s in low_act[:8]]
                print(f"  动态拉黑: {', '.join(names)}" + (f' ...等{len(low_act)}只' if len(low_act) > 8 else ''))
            if activity_filter.resurrected:
                print(f"  放量复活: {', '.join(sorted(activity_filter.resurrected))}")

        if len(signals) < 10:
            return signals, []

        # ── 分组 ──
        self.grouper = StockGrouper(stock_df, signals)
        self.grouper.assign_groups()
        if verbose:
            print(self.grouper.summary())

        survivors = signals[:]
        self.results = []

        # ── Stage 1: 粗筛 (300 → ~150, 高活性组多留, 低波组严筛) ──
        if verbose:
            print(f"\n{'='*50}")
            print(f"Stage 1 粗筛: {len(survivors)} 只 → 动态保留 + 收益下限过滤")
            print(f"{'='*50}")
        result1 = self._run_stage(1, survivors, stock_df, keep_ratio=0.50, verbose=verbose,
                                  return_floor=0.008,  # 预测收益低于0.8%直接淘汰
                                  group_weight_bias={'highvol': 1.5, 'midvol': 1.0, 'lowvol': 0.5})
        self.results.append(result1)
        survivors = result1.survivors
        if verbose:
            print(f"  Stage 1 存活: {len(survivors)} 只")

        # ── Stage 2: 精选 (~150 → ~45, 更严的收益下限) ──
        if verbose:
            print(f"\n{'='*50}")
            print(f"Stage 2 精选: {len(survivors)} 只 → 动态保留 + 收益下限过滤")
            print(f"{'='*50}")
        result2 = self._run_stage(2, survivors, stock_df, keep_ratio=0.30, verbose=verbose,
                                  return_floor=0.006,
                                  group_weight_bias={'highvol': 1.5, 'midvol': 1.0, 'lowvol': 0.5})
        self.results.append(result2)
        survivors = result2.survivors
        if verbose:
            print(f"  Stage 2 存活: {len(survivors)} 只")

        # ── 复活赛 ──
        all_eliminated = result1.eliminated + result2.eliminated + low_act  # 动态拉黑的也能参加复活
        # 去重
        seen = set()
        eliminated_unique = []
        for s in all_eliminated:
            if s.stock_id not in seen:
                seen.add(s.stock_id)
                eliminated_unique.append(s)

        if verbose:
            print(f"\n{'='*50}")
            print(f"复活赛: 淘汰池 {len(eliminated_unique)} 只 → TFT+TimesNet 救回 ≤5 只")
            print(f"{'='*50}")
        rescued = self.resurrection.evaluate(eliminated_unique, stock_df, max_rescue=5)

        # ── 决赛候选 ──
        final_candidates = list(survivors) + rescued
        # 去重
        seen2 = set()
        final_unique = []
        for s in final_candidates:
            if s.stock_id not in seen2:
                seen2.add(s.stock_id)
                final_unique.append(s)

        if verbose:
            print(f"\n[决赛池] {len(final_unique)} 只 (存活 {len(survivors)} + 复活 {len(rescued)})")

        return final_unique, self.results

    def _run_stage(self, stage_num, survivors, stock_df, keep_ratio=0.5, verbose=True,
                   return_floor=0.0, group_weight_bias=None):
        """执行一个筛选阶段

        核心逻辑:
        1. 按分组分别打分（每组用不同的模型组合）
        2. 组内 z-score 标准化排名
        3. 收益下限过滤: 预测收益 < return_floor 直接淘汰
        4. 高活性组倾斜: highvol 组多留, lowvol 组严筛
        5. 跨组时 z-score 可比
        """
        if group_weight_bias is None:
            group_weight_bias = {}

        group_signals = defaultdict(list)
        for s in survivors:
            g = self.grouper.get_group(s.stock_id)
            group_signals[g].append(s)

        all_survivors = []
        all_eliminated = []
        all_scores = {}
        n_floor_eliminated = 0
        n_ratio_eliminated = 0

        n_groups = len(group_signals)
        for group, signals_in_group in tqdm(list(group_signals.items()), desc=f"  Stage {stage_num} 各组筛选", unit="组", leave=False, total=n_groups):
            if len(signals_in_group) < 3:
                all_survivors.extend(signals_in_group)
                continue

            # 获取本阶段该组的模型
            model_names = StageConfig.get_models(stage_num, group)

            # 用指定模型打分
            scored = []
            for signal in signals_in_group:
                score = self._score_with_models(signal, stock_df, model_names)
                scored.append((signal, score))

            # 组内 z-score 标准化
            scores_arr = np.array([sc for _, sc in scored])
            mean_s = scores_arr.mean()
            std_s = scores_arr.std() + 1e-8
            z_scored = [(sig, (sc - mean_s) / std_s) for sig, sc in scored]

            # 按 z-score 降序排列
            z_scored.sort(key=lambda x: x[1], reverse=True)

            # ---- 动态保留比例: 高活性组多留 ----
            vol_type = group.split('_')[0]  # highvol / midvol / lowvol
            bias = group_weight_bias.get(vol_type, 1.0)
            effective_ratio = np.clip(keep_ratio * bias, 0.25, 0.75)
            n_keep = max(3, int(len(z_scored) * effective_ratio))

            # ---- 收益下限过滤: 预测收益不够的直接淘汰 ----
            group_survivors = []
            group_eliminated = []
            for sig, z in z_scored:
                if sig.predicted_return < return_floor:
                    # 收益不达标，淘汰
                    group_eliminated.append(sig)
                    n_floor_eliminated += 1
                elif len(group_survivors) < n_keep:
                    group_survivors.append(sig)
                else:
                    group_eliminated.append(sig)
                    n_ratio_eliminated += 1

            all_survivors.extend(group_survivors)
            all_eliminated.extend(group_eliminated)
            all_scores[group] = z_scored

            if verbose:
                best = z_scored[0]
                avg_ret = np.mean([s.predicted_return for s in signals_in_group])
                print(f"  {group}(avg_ret={avg_ret:.1%}): {len(signals_in_group)}→{len(group_survivors)} "
                      f"(bias={bias:.1f}x) | top={best[0].stock_id} z={best[1]:.3f}")

        if verbose and n_floor_eliminated > 0:
            print(f"  收益下限过滤淘汰: {n_floor_eliminated} 只 (pred_return < {return_floor:.1%})")

        return StageResult(
            stage=stage_num,
            survivors=all_survivors,
            eliminated=all_eliminated,
            group_scores=all_scores,
        )

    def _score_with_models(self, signal, stock_df, model_names):
        """用指定模型组合给单只股票打分

        Returns:
            综合得分（模型预测均值）
        """
        stock_data = stock_df[
            stock_df['stock_id'].astype(str).str.zfill(6) == signal.stock_id
        ].sort_values('date').set_index('date')

        if len(stock_data) < self.seq_len:
            return signal.predicted_return  # 数据不足，用原始预测

        features = self.fe.build_all_features(stock_data, self.industry_df, self.macro_df)
        features = self.fe.remove_outliers(features)
        last_features = features.iloc[-self.seq_len:].fillna(0)
        X = last_features.values.astype(np.float32)

        preds = []
        for model_name in model_names:
            try:
                model = self.predictor.base_models.get(model_name)
                if model is None:
                    continue

                if model_name in ['lightgbm', 'catboost', 'xgboost', 'spectralm']:
                    # GBDT: 用最后一行特征
                    X_scaled = self.predictor.scaler.transform(
                        X[-1:].reshape(1, -1)
                    ) if self.predictor.scaler else X[-1:].reshape(1, -1)
                    p = float(model.predict(X_scaled)[0])
                else:
                    # PyTorch: 用完整时序
                    X_t = np.expand_dims(X, axis=0).astype(np.float32)
                    if self.predictor.scaler:
                        X_t = np.array([
                            self.predictor.scaler.transform(X_t[0])
                        ])
                    import torch
                    with torch.no_grad():
                        t = torch.from_numpy(X_t).to(self.predictor.device)
                        p = float(model(t).cpu().numpy())
                preds.append(p)
            except Exception:
                continue

        if not preds:
            return signal.predicted_return

        return float(np.mean(preds))
