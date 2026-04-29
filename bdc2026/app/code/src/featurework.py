"""
特征工程模块 - 8大类120个特征 (修复版)
"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)


def set_all_seeds(seed=42):
    np.random.seed(seed)


set_all_seeds(42)


class FeatureEngineering:
    """多因子特征工程类"""

    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_list = []
        self.stock_list = []

    def calculate_basic_features(self, df):
        """量价基础特征（24个）"""
        features = pd.DataFrame(index=df.index)

        close = df['close']
        open_p = df['open']
        high = df['high']
        low = df['low']
        volume = df['volume']
        amount = df['amount']

        # 多周期收益率
        features['daily_return'] = close.pct_change()
        features['return_5d'] = close.pct_change(5)
        features['return_10d'] = close.pct_change(10)
        features['return_20d'] = close.pct_change(20)
        features['return_60d'] = close.pct_change(60)
        features['return_120d'] = close.pct_change(120)

        # 振幅/波动
        features['amplitude'] = (high - low) / (close + 1e-8)
        features['turnover_rate'] = df['turnover_rate'] if 'turnover_rate' in df.columns else 0
        features['log_amount'] = np.log1p(amount)
        features['volume_change'] = volume.pct_change()
        features['high_low_ratio'] = high / (low + 1e-8)
        features['open_close_ratio'] = open_p / (close + 1e-8)
        features['intraday_volatility'] = (high - low) / (open_p + 1e-8)

        # 量比（缓存rolling结果避免重复计算）
        vol_ma5 = volume.rolling(5).mean()
        vol_ma10 = volume.rolling(10).mean()
        vol_ma20 = volume.rolling(20).mean()
        vol_ma60 = volume.rolling(60).mean()
        amt_ma5 = amount.rolling(5).mean()
        amt_ma10 = amount.rolling(10).mean()
        amt_ma20 = amount.rolling(20).mean()
        amt_ma60 = amount.rolling(60).mean()

        features['volume_ma_ratio_5'] = volume / (vol_ma5 + 1e-8)
        features['volume_ma_ratio_10'] = volume / (vol_ma10 + 1e-8)
        features['volume_ma_ratio_20'] = volume / (vol_ma20 + 1e-8)
        features['volume_ma_ratio_60'] = volume / (vol_ma60 + 1e-8)
        features['amount_ma_ratio_5'] = amount / (amt_ma5 + 1e-8)
        features['amount_ma_ratio_10'] = amount / (amt_ma10 + 1e-8)
        features['amount_ma_ratio_20'] = amount / (amt_ma20 + 1e-8)
        features['amount_ma_ratio_60'] = amount / (amt_ma60 + 1e-8)

        # 动量
        features['price_momentum_5'] = close / (close.shift(5) + 1e-8) - 1
        features['price_momentum_10'] = close / (close.shift(10) + 1e-8) - 1
        features['price_momentum_20'] = close / (close.shift(20) + 1e-8) - 1

        # 非线性交互
        features['price_volume_interaction'] = close * volume
        features['return_volume_interaction'] = features['daily_return'] * volume
        features['amplitude_volume_interaction'] = features['amplitude'] * volume

        return features

    def calculate_technical_indicators(self, df):
        """技术指标特征（35个）"""
        features = pd.DataFrame(index=df.index)

        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']

        # 移动平均
        ma_5 = close.rolling(5).mean()
        ma_10 = close.rolling(10).mean()
        ma_20 = close.rolling(20).mean()
        ma_60 = close.rolling(60).mean()
        ma_120 = close.rolling(120).mean()

        features['ma_5'] = ma_5
        features['ma_10'] = ma_10
        features['ma_20'] = ma_20
        features['ma_60'] = ma_60
        features['ma_120'] = ma_120
        features['ma_ratio_5'] = close / (ma_5 + 1e-8)
        features['ma_ratio_10'] = close / (ma_10 + 1e-8)
        features['ma_ratio_20'] = close / (ma_20 + 1e-8)
        features['ma_ratio_60'] = close / (ma_60 + 1e-8)
        features['ma_ratio_120'] = close / (ma_120 + 1e-8)

        # MACD
        exp12 = close.ewm(span=12, adjust=False).mean()
        exp26 = close.ewm(span=26, adjust=False).mean()
        macd = exp12 - exp26
        signal = macd.ewm(span=9, adjust=False).mean()
        features['macd'] = macd
        features['macd_signal'] = signal
        features['macd_hist'] = macd - signal

        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        for period in [6, 12, 24]:
            avg_gain = gain.rolling(period).mean()
            avg_loss = loss.rolling(period).mean()
            features[f'rsi_{period}'] = 100 - (100 / (1 + avg_gain / (avg_loss + 1e-8)))

        # KDJ
        low_9 = low.rolling(9).min()
        high_9 = high.rolling(9).max()
        k_raw = 100 * (close - low_9) / (high_9 - low_9 + 1e-8)
        features['kdj_k'] = k_raw.rolling(3).mean()
        features['kdj_d'] = features['kdj_k'].rolling(3).mean()
        features['kdj_j'] = 3 * features['kdj_k'] - 2 * features['kdj_d']

        # 布林带
        bb_std = close.rolling(20).std()
        features['bb_middle'] = ma_20
        features['bb_upper'] = ma_20 + 2 * bb_std
        features['bb_lower'] = ma_20 - 2 * bb_std
        features['bb_width'] = (features['bb_upper'] - features['bb_lower']) / (ma_20 + 1e-8)
        features['bb_position'] = (close - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'] + 1e-8)

        # ATR
        hl = high - low
        hc = (high - close.shift()).abs()
        lc = (low - close.shift()).abs()
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        features['atr_14'] = tr.rolling(14).mean()

        # OBV
        features['obv'] = (np.sign(close.diff()) * volume).cumsum()
        features['obv_ma'] = features['obv'].rolling(10).mean()

        # Williams %R
        h14 = high.rolling(14).max()
        l14 = low.rolling(14).min()
        features['williams_r'] = -100 * (h14 - close) / (h14 - l14 + 1e-8)

        # CCI
        tp = (high + low + close) / 3
        tp_ma14 = tp.rolling(14).mean()
        tp_std14 = tp.rolling(14).std()
        features['cci_14'] = (tp - tp_ma14) / (0.015 * tp_std14 + 1e-8)

        # ROC
        features['roc_12'] = (close - close.shift(12)) / (close.shift(12) + 1e-8) * 100
        features['roc_24'] = (close - close.shift(24)) / (close.shift(24) + 1e-8) * 100

        # DMI
        up_move = high.diff()
        down_move = -low.diff()
        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
        atr14 = tr.rolling(14).mean()
        features['dmi_plus'] = 100 * plus_dm.rolling(14).mean() / (atr14 + 1e-8)
        features['dmi_minus'] = 100 * minus_dm.rolling(14).mean() / (atr14 + 1e-8)

        # BIAS
        features['bias_5'] = (close - ma_5) / (ma_5 + 1e-8) * 100
        features['bias_10'] = (close - ma_10) / (ma_10 + 1e-8) * 100
        features['bias_20'] = (close - ma_20) / (ma_20 + 1e-8) * 100

        return features

    def calculate_money_flow_features(self, df):
        """资金流特征（17个）"""
        features = pd.DataFrame(index=df.index)

        amount = df['amount']
        volume = df['volume']
        close = df['close']

        # 用日内价格位置估算买卖压力 (Close Location Value)
        high, low = df['high'], df['low']
        clv = ((close - low) - (high - close)) / (high - low + 1e-8)  # -1到1
        features['money_flow'] = clv * amount  # Chaikin Money Flow概念

        # 大/中/小单拆分（基于每日价格行为的合理代理）
        features['big_flow'] = features['money_flow'] * 0.5
        features['medium_flow'] = features['money_flow'] * 0.3
        features['small_flow'] = features['money_flow'] * 0.2

        features['big_flow_ratio'] = features['big_flow'] / (amount + 1e-8)
        features['medium_flow_ratio'] = features['medium_flow'] / (amount + 1e-8)
        features['small_flow_ratio'] = features['small_flow'] / (amount + 1e-8)

        # 资金流趋势（滚动对比）
        bf = features['big_flow']
        features['flow_trend_short'] = bf.rolling(5).mean() / (bf.rolling(20).mean() + 1e-8)
        features['flow_trend_long'] = bf.rolling(10).mean() / (bf.rolling(60).mean() + 1e-8)

        # 资金流Z-score
        bf_mean = bf.rolling(20).mean()
        bf_std = bf.rolling(20).std()
        features['flow_zscore'] = (bf - bf_mean) / (bf_std + 1e-8)

        # 资金流与价格相关性
        price_change = close.pct_change()
        features['flow_return_corr'] = price_change.rolling(20).corr(bf / (volume + 1))

        # 资金流移动平均
        for w in [5, 10, 20]:
            features[f'flow_ma_{w}'] = bf.rolling(w).mean()
            features[f'flow_ma_ratio_{w}'] = bf / (features[f'flow_ma_{w}'] + 1e-8)

        return features

    def calculate_industry_macro_features(self, df, industry_df=None, macro_df=None):
        """行业与宏观特征（15个）"""
        features = pd.DataFrame(index=df.index)
        close = df['close']

        # 如果提供了行业数据，按日期join
        if industry_df is not None and len(industry_df) > 0:
            # 取行业平均值作为市场代理
            ind_daily = industry_df.groupby('date').agg({
                'industry_return': 'mean',
                'industry_pe': 'mean',
                'industry_pb': 'mean',
                'industry_turnover': 'mean',
            }).reset_index()
            ind_daily['date'] = pd.to_datetime(ind_daily['date'])

            # 将行业数据合并到df上（需要先reset_index处理date）
            df_with_date = df.reset_index() if 'date' in (df.index.name or '') else df.copy()
            if 'date' not in df_with_date.columns:
                df_with_date = df.copy()
                df_with_date['date'] = df_with_date.index

            # 对齐日期
            ind_daily = ind_daily.set_index('date')
            for col in ['industry_return', 'industry_pe', 'industry_pb', 'industry_turnover']:
                if col in ind_daily.columns:
                    # 按index日期对齐
                    aligned = ind_daily[col].reindex(df.index, method='ffill')
                    features[col] = aligned.values if hasattr(aligned, 'values') else aligned
                else:
                    features[col] = 0

        if 'industry_return' not in features.columns:
            features['industry_return'] = 0
        if 'industry_pe' not in features.columns:
            features['industry_pe'] = 0
        if 'industry_pb' not in features.columns:
            features['industry_pb'] = 0
        if 'industry_turnover' not in features.columns:
            features['industry_turnover'] = 0

        # 相对行业/市场
        features['return_vs_industry'] = close.pct_change() - features['industry_return']
        features['industry_pe_percentile'] = features['industry_pe']
        features['industry_pb_percentile'] = features['industry_pb']

        # 宏观特征（兼容AKShare真实数据和模拟数据，确保列数固定）
        # 所有可能的宏观源列（AKShare + 模拟数据全集），缺失列填0保证维度一致
        ALL_MACRO_SOURCE_COLS = ['cpi', 'ppi', 'pmi', 'pmi_nonmanufacturing',
                                 'm2', 'm2_yoy', 'lpr_1y', 'lpr_5y', 'usdcny']
        if macro_df is not None and len(macro_df) > 0:
            macro_df = macro_df.copy()
            macro_df['date'] = pd.to_datetime(macro_df['date'])
            macro_df = macro_df.set_index('date')
            # 补全缺失的宏观列（API偶发失败导致列缺失，补0保证维度一致）
            for col in ALL_MACRO_SOURCE_COLS:
                if col not in macro_df.columns:
                    macro_df[col] = 0.0
            # 统一映射
            macro_col_map = [
                ('cpi', 'cpi'), ('ppi', 'ppi'), ('pmi', 'pmi'),
                ('m2', 'm2'), ('usdcny', 'usdcny'),
                ('m2_yoy', 'm2_yoy'),
                ('lpr_1y', 'lpr_1y'), ('lpr_5y', 'lpr_5y'),
                ('pmi_nonmanufacturing', 'pmi_nonmanufacturing'),
            ]
            for src_col, feat_name in macro_col_map:
                if src_col in macro_df.columns:
                    aligned = macro_df[src_col].reindex(df.index, method='ffill')
                    features[f'macro_{feat_name}'] = aligned.values if hasattr(aligned, 'values') else aligned
                elif src_col in ALL_MACRO_SOURCE_COLS:
                    features[f'macro_{feat_name}'] = 0
        else:
            # 无宏观数据时填0保证维度一致
            for col in ALL_MACRO_SOURCE_COLS:
                features[f'macro_{col}'] = 0

        for col in ['macro_cpi', 'macro_ppi', 'macro_m2']:
            if col in features.columns:
                features[f'{col}_change'] = features[col].pct_change()
        if 'macro_pmi' in features.columns:
            features['pmi_change'] = features['macro_pmi'].diff()
        if 'macro_bond_10y' in features.columns:
            features['interest_rate_change'] = features['macro_bond_10y'].diff()
        # AKShare额外宏观特征的变化率
        if 'macro_m2_yoy' in features.columns:
            features['m2_yoy_change'] = features['macro_m2_yoy'].diff()
        if 'macro_lpr_1y' in features.columns:
            features['lpr_change'] = features['macro_lpr_1y'].diff()
        if 'macro_pmi_nonmanufacturing' in features.columns:
            features['pmi_nonmfg_change'] = features['macro_pmi_nonmanufacturing'].diff()

        return features

    def calculate_time_series_features(self, df):
        """时序统计特征（15个）"""
        features = pd.DataFrame(index=df.index)
        close = df['close']
        returns = close.pct_change()

        for window in [5, 10, 20, 60]:
            ret_std = returns.rolling(window).std()
            features[f'return_std_{window}'] = ret_std
            features[f'return_skew_{window}'] = returns.rolling(window).skew()
            features[f'return_kurt_{window}'] = returns.rolling(window).kurt()

        # 最大回撤：当前价格相对窗口内最高价的回撤（每行不同）
        for window in [20, 60]:
            rolling_max = close.rolling(window).max()
            features[f'max_drawdown_{window}'] = (close - rolling_max) / (rolling_max + 1e-8)

        # 滚动夏普比率
        for window in [20, 60]:
            mean_ret = returns.rolling(window).mean()
            std_ret = returns.rolling(window).std()
            features[f'sharp_ratio_{window}'] = mean_ret / (std_ret + 1e-8) * np.sqrt(252)

        # 动量和反转
        features['momentum_20_60'] = close / (close.shift(60) + 1e-8) - close / (close.shift(20) + 1e-8)
        features['reversal_5_20'] = close / (close.shift(20) + 1e-8) - close / (close.shift(5) + 1e-8)
        features['volatility_change'] = returns.rolling(20).std() / (returns.rolling(60).std() + 1e-8)

        return features

    def calculate_valuation_features(self, df):
        """估值特征（5个）"""
        features = pd.DataFrame(index=df.index)

        pe = df.get('pe', pd.Series(np.nan, index=df.index)).fillna(15)
        pb = df.get('pb', pd.Series(np.nan, index=df.index)).fillna(1.5)
        close = df['close']

        pe_rolling_min = pe.rolling(252 * 3, min_periods=60).min()
        pe_rolling_max = pe.rolling(252 * 3, min_periods=60).max()
        features['pe_percentile'] = (pe - pe_rolling_min) / (pe_rolling_max - pe_rolling_min + 1e-8)

        pb_rolling_min = pb.rolling(252 * 3, min_periods=60).min()
        pb_rolling_max = pb.rolling(252 * 3, min_periods=60).max()
        features['pb_percentile'] = (pb - pb_rolling_min) / (pb_rolling_max - pb_rolling_min + 1e-8)

        features['peg'] = pe / (close.pct_change(252) * 100 + 1e-8)
        features['ps_ratio'] = close / (df.get('revenue_per_share', pd.Series(1, index=df.index)) + 1e-8)
        features['pcf_ratio'] = close / (df.get('cashflow_per_share', pd.Series(1, index=df.index)) + 1e-8)

        return features

    def calculate_liquidity_features(self, df):
        """流动性特征（3个）"""
        features = pd.DataFrame(index=df.index)
        amount = df['amount']
        close = df['close']

        features['avg_daily_amount_20'] = amount.rolling(20).mean()
        features['avg_turnover_20'] = df['turnover_rate'].rolling(20).mean() if 'turnover_rate' in df.columns else amount.rolling(20).mean() * 0
        features['bid_ask_spread'] = (df['high'] - df['low']) / (close + 1e-8)

        return features

    def calculate_event_features(self, df):
        """事件特征（2个）"""
        features = pd.DataFrame(index=df.index)
        features['index_adjustment'] = df.get('is_in_index', pd.Series(1, index=df.index))
        features['earnings_season'] = df.get('earnings_season', pd.Series(0, index=df.index))
        return features

    def build_all_features(self, stock_df, industry_df=None, macro_df=None):
        """构建所有特征"""
        set_all_seeds(42)

        all_components = [
            self.calculate_basic_features(stock_df),
            self.calculate_technical_indicators(stock_df),
            self.calculate_money_flow_features(stock_df),
            self.calculate_industry_macro_features(stock_df, industry_df, macro_df),
            self.calculate_time_series_features(stock_df),
            self.calculate_valuation_features(stock_df),
            self.calculate_liquidity_features(stock_df),
            self.calculate_event_features(stock_df),
        ]

        result = pd.concat(all_components, axis=1)
        result = result.replace([np.inf, -np.inf], np.nan)
        result = result.ffill().bfill().fillna(0)

        return result

    def remove_outliers(self, features, n_std=3):
        """去除极端值（3σ截断）"""
        for col in features.columns:
            mean = features[col].mean()
            std = features[col].std()
            if std > 0:
                features[col] = features[col].clip(mean - n_std * std, mean + n_std * std)
        return features

    def standardize_features(self, features, fit=True):
        """标准化特征"""
        if fit:
            self.scaler.fit(features)
        return pd.DataFrame(self.scaler.transform(features), index=features.index, columns=features.columns)

    def select_top_features(self, features, target, n_features=100):
        """使用LightGBM特征重要性筛选Top N特征"""
        try:
            import lightgbm as lgb
            set_all_seeds(42)

            valid_idx = features.dropna().index.intersection(target.dropna().index)
            X = features.loc[valid_idx].fillna(0)
            y = target.loc[valid_idx]

            if len(X) < 100:
                return features.columns.tolist()[:min(len(features.columns), n_features)]

            model = lgb.LGBMRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42,
                verbose=-1
            )
            model.fit(X, y)

            importance = pd.Series(model.feature_importances_, index=features.columns)
            top_features = importance.nlargest(n_features).index.tolist()
            return top_features
        except Exception as e:
            print(f"Feature selection failed: {e}")
            return features.columns.tolist()[:min(len(features.columns), n_features)]

    def remove_correlated_features(self, features, threshold=0.9):
        """去除高度相关特征（相关系数>threshold）"""
        # 采样计算避免大数据OOM
        if len(features) > 50000:
            sample_idx = np.random.choice(len(features), 50000, replace=False)
            corr_matrix = features.iloc[sample_idx].corr().abs()
        else:
            corr_matrix = features.corr().abs()

        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
        features = features.drop(columns=to_drop)
        return features, to_drop


def generate_features_for_stocks(stock_data_dict, industry_data=None, macro_data=None):
    """为多只股票生成特征"""
    all_stock_features = {}
    all_targets = {}

    for stock_id, df in stock_data_dict.items():
        fe = FeatureEngineering()
        features = fe.build_all_features(df, industry_data, macro_data)
        features = fe.remove_outliers(features)

        close = df['close']
        open_p = df['open']
        target = close.shift(-5) / (open_p.shift(-1) + 1e-8) - 1

        all_stock_features[stock_id] = features
        all_targets[stock_id] = target

    return all_stock_features, all_targets


if __name__ == "__main__":
    print("特征工程模块测试")
    dates = pd.date_range('2020-01-01', periods=500)
    test_data = pd.DataFrame({
        'open': np.random.randn(500).cumsum() + 100,
        'high': np.random.randn(500).cumsum() + 103,
        'low': np.random.randn(500).cumsum() + 97,
        'close': np.random.randn(500).cumsum() + 100,
        'volume': np.random.randint(1000000, 10000000, 500),
        'amount': np.random.randint(100000000, 1000000000, 500),
        'turnover_rate': np.random.rand(500) * 5,
        'pe': np.random.rand(500) * 30 + 10,
        'pb': np.random.rand(500) * 2 + 0.5,
    }, index=dates)

    fe = FeatureEngineering()
    features = fe.build_all_features(test_data)
    print(f"生成特征数量: {len(features.columns)}")
    print(f"特征列: {features.columns.tolist()[:10]}...")
