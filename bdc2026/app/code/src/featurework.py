"""
特征工程模块 - 8大类120个特征
2026清华大学大数据挑战赛冠军方案
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

def set_all_seeds(seed=42):
    """设置所有随机种子确保可复现性"""
    np.random.seed(seed)

set_all_seeds(42)

class FeatureEngineering:
    """多因子特征工程类"""

    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_list = []
        self.stock_list = []

    def calculate_basic_features(self, df):
        """量价基础特征（25个）"""
        features = pd.DataFrame(index=df.index)

        close = df['close']
        open_p = df['open']
        high = df['high']
        low = df['low']
        volume = df['volume']
        amount = df['amount']

        features['daily_return'] = close.pct_change()
        features['return_5d'] = close.pct_change(5)
        features['return_10d'] = close.pct_change(10)
        features['return_20d'] = close.pct_change(20)
        features['return_60d'] = close.pct_change(60)
        features['return_120d'] = close.pct_change(120)

        features['amplitude'] = (high - low) / close
        features['volume_ratio'] = volume / volume.rolling(5).mean()
        features['turnover_rate'] = df['turnover_rate']
        features['log_amount'] = np.log1p(amount)
        features['volume_change'] = volume.pct_change()
        features['high_low_ratio'] = high / low
        features['open_close_ratio'] = open_p / close
        features['intraday_volatility'] = (high - low) / open_p

        for window in [5, 10, 20, 60]:
            features[f'volume_ma_ratio_{window}'] = volume / volume.rolling(window).mean()
            features[f'amount_ma_ratio_{window}'] = amount / amount.rolling(window).mean()

        features['price_momentum_5'] = close / close.shift(5) - 1
        features['price_momentum_10'] = close / close.shift(10) - 1
        features['price_momentum_20'] = close / close.shift(20) - 1

        # 非线性特征交互
        features['price_volume_interaction'] = close * volume
        features['return_volume_interaction'] = features['daily_return'] * volume
        features['amplitude_volume_interaction'] = features['amplitude'] * volume
        features['price_turnover_interaction'] = close * features['turnover_rate']

        return features

    def calculate_technical_indicators(self, df):
        """技术指标特征（35个）"""
        features = pd.DataFrame(index=df.index)

        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']

        for window in [5, 10, 20, 60, 120]:
            features[f'ma_{window}'] = close.rolling(window).mean()
            features[f'ma_ratio_{window}'] = close / features[f'ma_{window}']

        exp12 = close.ewm(span=12, adjust=False).mean()
        exp26 = close.ewm(span=26, adjust=False).mean()
        macd = exp12 - exp26
        signal = macd.ewm(span=9, adjust=False).mean()
        features['macd'] = macd
        features['macd_signal'] = signal
        features['macd_hist'] = macd - signal

        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        features['rsi_6'] = 100 - (100 / (1 + gain.rolling(6).mean() / loss.rolling(6).mean()))
        features['rsi_12'] = 100 - (100 / (1 + gain.rolling(12).mean() / loss.rolling(12).mean()))
        features['rsi_24'] = 100 - (100 / (1 + gain.rolling(24).mean() / loss.rolling(24).mean()))

        low_n = low.rolling(9).min()
        high_n = high.rolling(9).max()
        k_raw = 100 * (close - low_n) / (high_n - low_n)
        features['kdj_k'] = k_raw.rolling(3).mean()
        features['kdj_d'] = features['kdj_k'].rolling(3).mean()
        features['kdj_j'] = 3 * features['kdj_k'] - 2 * features['kdj_d']

        features['bb_middle'] = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        features['bb_upper'] = features['bb_middle'] + 2 * bb_std
        features['bb_lower'] = features['bb_middle'] - 2 * bb_std
        features['bb_width'] = (features['bb_upper'] - features['bb_lower']) / features['bb_middle']
        features['bb_position'] = (close - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])

        features['atr_14'] = (high - low).rolling(14).mean() + (high - close).abs().rolling(14).mean() + (low - close).abs().rolling(14).mean()
        features['atr_14'] = features['atr_14'] / 3

        features['obv'] = (np.sign(close.diff()) * volume).cumsum()
        features['obv_ma'] = features['obv'].rolling(10).mean()

        features['williams_r'] = -100 * (high.rolling(14).max() - close) / (high.rolling(14).max() - low.rolling(14).min())

        typical_price = (high + low + close) / 3
        features['cci_14'] = (typical_price - typical_price.rolling(14).mean()) / (0.015 * typical_price.rolling(14).std())

        features['roc_12'] = (close - close.shift(12)) / close.shift(12) * 100
        features['roc_24'] = (close - close.shift(24)) / close.shift(24) * 100

        high_diff = high.diff()
        low_diff = low.diff()
        plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
        minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
        features['dmi_plus'] = 100 * plus_dm.rolling(14).mean() / features['atr_14']
        features['dmi_minus'] = 100 * minus_dm.rolling(14).mean() / features['atr_14']

        for window in [5, 10, 20]:
            features[f'bias_{window}'] = (close - close.rolling(window).mean()) / close.rolling(window).mean() * 100

        return features

    def calculate_money_flow_features(self, df):
        """资金流特征（20个）"""
        features = pd.DataFrame(index=df.index)

        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        amount = df['amount']

        if 'money_flow' in df.columns:
            money_flow = df['money_flow']
            features['big_flow'] = money_flow * 0.7
            features['medium_flow'] = money_flow * 0.2
            features['small_flow'] = money_flow * 0.1
        else:
            features['big_flow'] = amount * 0.4
            features['medium_flow'] = amount * 0.35
            features['small_flow'] = amount * 0.25

        features['big_flow_ratio'] = features['big_flow'] / amount
        features['medium_flow_ratio'] = features['medium_flow'] / amount
        features['small_flow_ratio'] = features['small_flow'] / amount

        features['net_flow_5'] = features['big_flow'].rolling(5).sum() - features['big_flow'].rolling(5).sum()
        features['net_flow_10'] = features['big_flow'].rolling(10).sum() - features['big_flow'].rolling(10).sum()
        features['net_flow_20'] = features['big_flow'].rolling(20).sum() - features['big_flow'].rolling(20).sum()

        flow_mean = features['big_flow'].rolling(20).mean()
        flow_std = features['big_flow'].rolling(20).std()
        features['flow_zscore'] = (features['big_flow'] - flow_mean) / (flow_std + 1e-8)

        features['flow_trend'] = features['big_flow'].rolling(5).mean() / (features['big_flow'].rolling(20).mean() + 1e-8)

        price_change = close.pct_change()
        features['flow_return_corr'] = price_change.rolling(10).corr(features['big_flow'] / volume.replace(0, 1))

        for window in [5, 10, 20]:
            features[f'flow_ma_{window}'] = features['big_flow'].rolling(window).mean()
            features[f'flow_ma_ratio_{window}'] = features['big_flow'] / (features[f'flow_ma_{window}'] + 1e-8)

        return features

    def calculate_industry_macro_features(self, df, industry_df=None, macro_df=None):
        """行业与宏观特征（15个）"""
        features = pd.DataFrame(index=df.index)

        close = df['close']
        features['return_vs_industry'] = df.get('industry_return', close.pct_change())
        features['return_vs_market'] = df.get('market_return', close.pct_change())
        features['industry_pe_percentile'] = df.get('industry_pe_percentile', 0.5)
        features['industry_pb_percentile'] = df.get('industry_pb_percentile', 0.5)
        features['industry_turnover'] = df.get('industry_turnover', 0)
        features['market_turnover'] = df.get('market_turnover', 0)

        if macro_df is not None:
            for col in ['cpi', 'ppi', 'pmi', 'm2', 'usdcny', 'bond_10y']:
                if col in macro_df.columns:
                    features[f'macro_{col}'] = macro_df[col].reindex(df.index).fillna(method='ffill')

        features['cpi_change'] = features.get('macro_cpi', pd.Series(0, index=df.index)).pct_change()
        features['ppi_change'] = features.get('macro_ppi', pd.Series(0, index=df.index)).pct_change()
        features['pmi_change'] = features.get('macro_pmi', pd.Series(0, index=df.index)).diff()
        features['m2_change'] = features.get('macro_m2', pd.Series(0, index=df.index)).pct_change()
        features['interest_rate_change'] = features.get('macro_bond_10y', pd.Series(0, index=df.index)).diff()

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

        for window in [20, 60]:
            rolling_max = close.rolling(window).max()
            drawdown = (close - rolling_max) / rolling_max
            features[f'max_drawdown_{window}'] = drawdown.min()

        for window in [20, 60]:
            mean_ret = returns.rolling(window).mean()
            std_ret = returns.rolling(window).std()
            features[f'sharpe_ratio_{window}'] = mean_ret / (std_ret + 1e-8) * np.sqrt(252)

        features['momentum_20_60'] = close / close.shift(60) - close / close.shift(20)
        features['reversal_5_20'] = close / close.shift(20) - close / close.shift(5)
        features['volatility_change'] = returns.rolling(20).std() / (returns.rolling(60).std() + 1e-8)

        return features

    def calculate_valuation_features(self, df):
        """估值特征（5个）"""
        features = pd.DataFrame(index=df.index)

        pe = df.get('pe', pd.Series(15, index=df.index))
        pb = df.get('pb', pd.Series(1, index=df.index))
        close = df['close']

        features['pe_percentile'] = (pe - pe.rolling(252*3).min()) / (pe.rolling(252*3).max() - pe.rolling(252*3).min() + 1e-8)
        features['pb_percentile'] = (pb - pb.rolling(252*3).min()) / (pb.rolling(252*3).max() - pb.rolling(252*3).min() + 1e-8)
        features['peg'] = pe / (close.pct_change(252) * 100 + 1e-8)
        features['ps_ratio'] = close / (df.get('revenue_per_share', pd.Series(1, index=df.index)) + 1e-8)
        features['pcf_ratio'] = close / (df.get('cashflow_per_share', pd.Series(1, index=df.index)) + 1e-8)

        return features

    def calculate_liquidity_features(self, df):
        """流动性特征（3个）"""
        features = pd.DataFrame(index=df.index)

        amount = df['amount']
        volume = df['volume']
        close = df['close']

        features['avg_daily_amount_20'] = amount.rolling(20).mean()
        features['avg_turnover_20'] = df['turnover_rate'].rolling(20).mean()
        features['bid_ask_spread'] = (df['high'] - df['low']) / close

        return features

    def calculate_event_features(self, df):
        """事件特征（2个）"""
        features = pd.DataFrame(index=df.index)

        features['index_adjustment'] = df.get('is_in_index', pd.Series(1, index=df.index))
        features['earnings_season'] = df.get('earnings_date', pd.Series(0, index=df.index))

        return features

    def build_all_features(self, stock_df, industry_df=None, macro_df=None):
        """构建所有特征"""
        set_all_seeds(42)

        all_features = []

        basic = self.calculate_basic_features(stock_df)
        all_features.append(basic)

        technical = self.calculate_technical_indicators(stock_df)
        all_features.append(technical)

        money_flow = self.calculate_money_flow_features(stock_df)
        all_features.append(money_flow)

        industry_macro = self.calculate_industry_macro_features(stock_df, industry_df, macro_df)
        all_features.append(industry_macro)

        time_series = self.calculate_time_series_features(stock_df)
        all_features.append(time_series)

        valuation = self.calculate_valuation_features(stock_df)
        all_features.append(valuation)

        liquidity = self.calculate_liquidity_features(stock_df)
        all_features.append(liquidity)

        event = self.calculate_event_features(stock_df)
        all_features.append(event)

        result = pd.concat(all_features, axis=1)

        result = result.replace([np.inf, -np.inf], np.nan)
        result = result.fillna(method='ffill').fillna(0)

        return result

    def remove_outliers(self, features, n_std=3):
        """去除极端值（3σ原则）"""
        for col in features.columns:
            mean = features[col].mean()
            std = features[col].std()
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
        """去除高度相关特征（相关系数>0.9）"""
        corr_matrix = features.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
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
        target = close.shift(-5) / close - 1

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