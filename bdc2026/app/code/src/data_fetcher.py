"""
数据获取模块 - 多源数据融合 (修复版)
"""
import os
import numpy as np
import pandas as pd
from datetime import datetime

np.random.seed(42)

# ---- 配置常量 ----
LOOKBACK_DAYS = 365        # 动态1年数据窗口
MIN_LISTED_YEARS = 0.5       # 动态窗口下放宽至半年
MIN_DAILY_AMOUNT = 1e8


def download_baostock_data():
    """从baostock下载沪深300成分股历史数据（含PE/PB/turnover）"""
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

    # 获取沪深300成分股列表
    hs300 = bs.query_hs300_stocks()
    stock_codes = []
    while (hs300.error_code == '0') and (hs300.next()):
        stock_codes.append(hs300.get_row_data()[1])
    bs.logout()

    if not stock_codes:
        print("未获取到沪深300成分股列表")
        return None, None

    print(f"获取到 {len(stock_codes)} 只成分股，开始下载全部日线数据...")

    all_data = []
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - pd.Timedelta(days=LOOKBACK_DAYS)).strftime('%Y-%m-%d')
    print(f"数据窗口: {start_date} ~ {end_date} (动态{LOOKBACK_DAYS}天)")

    bs.login()
    for i, code in enumerate(stock_codes):
        # 解析baostock代码格式
        if code.startswith('sh.') or code.startswith('sz.'):
            bs_code = code
            clean_code = code.replace('sh.', '').replace('sz.', '')
        elif code.startswith('6'):
            bs_code = f"sh.{code}"
            clean_code = code
        else:
            bs_code = f"sz.{code}"
            clean_code = code

        rs = bs.query_history_k_data_plus(
            bs_code,
            "date,open,high,low,close,volume,amount,turn,peTTM,pbMRQ",
            start_date=start_date,
            end_date=end_date,
            frequency='d'
        )

        while rs.error_code == '0' and rs.next():
            row = rs.get_row_data()
            all_data.append({
                'stock_id': clean_code,
                'date': row[0],
                'open': float(row[1]) if row[1] and row[1] != '' else np.nan,
                'high': float(row[2]) if row[2] and row[2] != '' else np.nan,
                'low': float(row[3]) if row[3] and row[3] != '' else np.nan,
                'close': float(row[4]) if row[4] and row[4] != '' else np.nan,
                'volume': int(float(row[5])) if row[5] and row[5] != '' else 0,
                'amount': float(row[6]) if row[6] and row[6] != '' else 0.0,
                'turnover_rate': float(row[7]) if row[7] and row[7] != '' else 0.0,
                'pe': float(row[8]) if row[8] and row[8] != '' else np.nan,
                'pb': float(row[9]) if row[9] and row[9] != '' else np.nan,
            })

        if (i + 1) % 50 == 0:
            print(f"已下载 {i+1}/{len(stock_codes)} 只股票")

    # 下载指数数据
    index_rs = bs.query_history_k_data_plus(
        "sh.000300",
        "date,open,high,low,close,volume,amount,turn",
        start_date=start_date,
        end_date=end_date,
        frequency='d'
    )

    index_data = []
    while index_rs.error_code == '0' and index_rs.next():
        row = index_rs.get_row_data()
        index_data.append({
            'date': row[0],
            'index_close': float(row[4]) if row[4] and row[4] != '' else np.nan,
            'index_volume': int(float(row[5])) if row[5] and row[5] != '' else 0,
            'index_amount': float(row[6]) if row[6] and row[6] != '' else 0.0,
            'index_turnover': float(row[7]) if row[7] and row[7] != '' else 0.0,
        })

    bs.logout()

    stock_df = pd.DataFrame(all_data) if all_data else None
    index_df = pd.DataFrame(index_data) if index_data else None

    if stock_df is not None and len(stock_df) > 0:
        print(f"成功下载 {len(stock_df)} 条股票记录，{stock_df['stock_id'].nunique()} 只股票")
        stock_csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'stock_data.csv')
        os.makedirs(os.path.dirname(stock_csv_path), exist_ok=True)
        stock_df.to_csv(stock_csv_path, index=False)
        print(f"股票数据已保存到 {stock_csv_path}")

    if index_df is not None and len(index_df) > 0:
        print(f"成功下载 {len(index_df)} 条指数记录")
        index_csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'index_data.csv')
        os.makedirs(os.path.dirname(index_csv_path), exist_ok=True)
        index_df.to_csv(index_csv_path, index=False)
        print(f"指数数据已保存到 {index_csv_path}")

    return stock_df, index_df


def _parse_chinese_month(date_str):
    """解析中文月份格式如 '2008年03月份' → datetime"""
    import re
    match = re.search(r'(\d{4})\D+(\d{1,2})', str(date_str))
    if match:
        return pd.Timestamp(year=int(match.group(1)), month=int(match.group(2)), day=1)
    return pd.NaT


def download_akshare_macro_data():
    """从AKShare下载真实宏观经济数据（免费公开数据源）"""
    try:
        import akshare as ak
    except ImportError:
        print("akshare未安装，使用模拟宏观数据")
        return None

    print("从AKShare下载真实宏观经济数据...")
    macro_records = []
    success_count = 0

    # --- CPI 月度同比 ---
    try:
        cpi = ak.macro_china_cpi_monthly()
        for _, row in cpi.iterrows():
            date_str = row.iloc[1]  # date column
            val = row.iloc[2]       # value column
            try:
                d = pd.to_datetime(date_str)
                if pd.notna(d) and pd.notna(val):
                    macro_records.append({'date': d, 'indicator': 'cpi', 'value': float(val)})
            except (ValueError, TypeError):
                pass
        success_count += 1
        print("  CPI数据获取成功")
    except Exception as e:
        print(f"  CPI获取失败: {e}")

    # --- PPI 月度同比 ---
    try:
        ppi = ak.macro_china_ppi_yearly()
        for _, row in ppi.iterrows():
            date_str = row.iloc[1]
            val = row.iloc[2]
            try:
                d = pd.to_datetime(date_str)
                if pd.notna(d) and pd.notna(val):
                    macro_records.append({'date': d, 'indicator': 'ppi', 'value': float(val)})
            except (ValueError, TypeError):
                pass
        success_count += 1
        print("  PPI数据获取成功")
    except Exception as e:
        print(f"  PPI获取失败: {e}")

    # --- PMI ---
    try:
        pmi = ak.macro_china_pmi()
        for _, row in pmi.iterrows():
            d = _parse_chinese_month(row.iloc[0])
            val_manufacturing = row.iloc[1]
            val_nonmanufacturing = row.iloc[3]
            if pd.notna(d):
                if pd.notna(val_manufacturing):
                    macro_records.append({'date': d, 'indicator': 'pmi', 'value': float(val_manufacturing)})
                if pd.notna(val_nonmanufacturing):
                    macro_records.append({'date': d, 'indicator': 'pmi_nonmanufacturing', 'value': float(val_nonmanufacturing)})
        success_count += 1
        print("  PMI数据获取成功")
    except Exception as e:
        print(f"  PMI获取失败: {e}")

    # --- M2 ---
    try:
        m2 = ak.macro_china_money_supply()
        for _, row in m2.iterrows():
            d = _parse_chinese_month(row.iloc[0])
            m2_amount = row.iloc[1]    # M2余额(亿元)
            m2_yoy = row.iloc[2]       # M2同比增速
            if pd.notna(d):
                if pd.notna(m2_amount):
                    macro_records.append({'date': d, 'indicator': 'm2', 'value': float(m2_amount)})
                if pd.notna(m2_yoy):
                    macro_records.append({'date': d, 'indicator': 'm2_yoy', 'value': float(m2_yoy)})
        success_count += 1
        print("  M2数据获取成功")
    except Exception as e:
        print(f"  M2获取失败: {e}")

    # --- LPR (利率) ---
    try:
        lpr = ak.macro_china_lpr()
        for _, row in lpr.iterrows():
            d = pd.to_datetime(row['TRADE_DATE'])
            if pd.notna(d) and pd.notna(row['LPR1Y']):
                macro_records.append({'date': d, 'indicator': 'lpr_1y', 'value': float(row['LPR1Y'])})
            if pd.notna(d) and pd.notna(row['LPR5Y']):
                macro_records.append({'date': d, 'indicator': 'lpr_5y', 'value': float(row['LPR5Y'])})
        success_count += 1
        print("  LPR数据获取成功")
    except Exception as e:
        print(f"  LPR获取失败: {e}")

    # --- 汇率 USDCNY ---
    try:
        fx = ak.currency_boc_sina(symbol='美元')
        for _, row in fx.iterrows():
            d = pd.to_datetime(row.iloc[0])
            rate = row.iloc[4]  # 央行中间价
            if pd.notna(d) and pd.notna(rate):
                macro_records.append({'date': d, 'indicator': 'usdcny', 'value': float(rate)})
        success_count += 1
        print("  汇率数据获取成功")
    except Exception as e:
        print(f"  汇率获取失败: {e}")

    if success_count < 2:
        print(f"  仅成功获取 {success_count}/6 项宏观数据，回退到模拟数据")
        return None

    # 转换为宽表格式
    macro_df = pd.DataFrame(macro_records)
    macro_df = macro_df.pivot_table(index='date', columns='indicator', values='value', aggfunc='last')
    macro_df = macro_df.sort_index()

    # 填充到日频
    all_dates = pd.date_range(macro_df.index.min(), macro_df.index.max(), freq='D')
    macro_df = macro_df.reindex(all_dates)
    macro_df = macro_df.ffill()

    # 重命名列为统一格式
    macro_df = macro_df.rename(columns={
        'cpi': 'cpi', 'ppi': 'ppi', 'pmi': 'pmi',
        'pmi_nonmanufacturing': 'pmi_nonmanufacturing',
        'm2': 'm2', 'm2_yoy': 'm2_yoy',
        'lpr_1y': 'lpr_1y', 'lpr_5y': 'lpr_5y', 'usdcny': 'usdcny'
    })
    macro_df = macro_df.reset_index().rename(columns={'index': 'date'})

    print(f"  宏观数据覆盖范围: {macro_df['date'].min().date()} ~ {macro_df['date'].max().date()}")
    print(f"  包含指标: {[c for c in macro_df.columns if c != 'date']}")
    return macro_df


def generate_macro_data():
    """回退方案：生成模拟宏观经济数据"""
    print("生成宏观经济模拟数据（基于历史合理区间）...")
    start_date = (datetime.now() - pd.Timedelta(days=LOOKBACK_DAYS)).strftime('%Y-%m-%d')
    dates = pd.date_range(start_date, datetime.now().strftime('%Y-%m-%d'), freq='M')

    np.random.seed(42)
    macro_data = []

    cpi_val, ppi_val, pmi_val = 102.0, 98.0, 50.2
    m2_val, usdcny_val, bond_val = 1200000, 6.2, 4.0

    for i, date in enumerate(dates):
        year = date.year
        if year < 2020:
            cpi_val *= (1 + np.random.randn() * 0.0015 + 0.0017)
        elif year < 2023:
            cpi_val *= (1 + np.random.randn() * 0.002 + 0.001)
        else:
            cpi_val *= (1 + np.random.randn() * 0.0015 + 0.0005)
        cpi_val = np.clip(cpi_val, 99, 104)

        if year <= 2016:
            ppi_val *= (1 + np.random.randn() * 0.003 - 0.002)
        elif year <= 2019:
            ppi_val *= (1 + np.random.randn() * 0.002 + 0.003)
        elif year <= 2022:
            ppi_val *= (1 + np.random.randn() * 0.003 + 0.002)
        else:
            ppi_val *= (1 + np.random.randn() * 0.003 - 0.002)
        ppi_val = np.clip(ppi_val, 93, 110)

        pmi_val += np.random.randn() * 1.5
        pmi_val = np.clip(pmi_val, 47, 53)
        m2_val *= (1 + np.random.randn() * 0.004 + 0.007)
        m2_val = np.clip(m2_val, 800000, 3200000)

        if year < 2018:
            usdcny_val += np.random.randn() * 0.03
        elif year < 2021:
            usdcny_val += np.random.randn() * 0.04 - 0.01
        else:
            usdcny_val += np.random.randn() * 0.04 + 0.01
        usdcny_val = np.clip(usdcny_val, 6.1, 7.4)

        if year < 2020:
            bond_val += np.random.randn() * 0.06
        else:
            bond_val += np.random.randn() * 0.06 - 0.03
        bond_val = np.clip(bond_val, 1.5, 5.0)

        macro_data.append({
            'date': date,
            'cpi': round(cpi_val, 2),
            'ppi': round(ppi_val, 2),
            'pmi': round(pmi_val, 2),
            'm2': round(m2_val, 0),
            'usdcny': round(usdcny_val, 4),
            'bond_10y': round(bond_val, 4),
        })

    return pd.DataFrame(macro_data)


def generate_industry_data():
    """生成申万行业指数模拟数据"""
    print("生成申万行业指数模拟数据...")
    start_date = (datetime.now() - pd.Timedelta(days=LOOKBACK_DAYS)).strftime('%Y-%m-%d')
    dates = pd.date_range(start_date, datetime.now().strftime('%Y-%m-%d'), freq='B')

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
        base_pe = np.random.uniform(10, 35)
        base_pb = np.random.uniform(0.8, 4.0)

        for date in dates:
            ret = np.random.randn() * 0.015 + 0.0002  # 微正漂移
            base_value *= (1 + ret)
            base_pe *= (1 + np.random.randn() * 0.008)
            base_pe = np.clip(base_pe, 5, 80)
            base_pb *= (1 + np.random.randn() * 0.008)
            base_pb = np.clip(base_pb, 0.3, 8)

            all_industry_data.append({
                'date': date,
                'industry': industry,
                'industry_return': ret,
                'industry_value': base_value,
                'industry_pe': base_pe,
                'industry_pb': base_pb,
                'industry_turnover': np.random.uniform(1, 8),
            })

    return pd.DataFrame(all_industry_data)


def preprocess_stock_data(df, min_listed_years=MIN_LISTED_YEARS):
    """数据预处理：统一格式、前向填充、删除上市不足N年的股票和流动性差的股票"""
    print("数据预处理...")

    if df is None or len(df) == 0:
        return df

    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['stock_id', 'date'])

    # 按股票前向填充
    fill_cols = ['open', 'high', 'low', 'close', 'volume', 'amount', 'turnover_rate', 'pe', 'pb']
    for col in fill_cols:
        if col in df.columns:
            df[col] = df.groupby('stock_id')[col].transform(lambda x: x.ffill().bfill())

    # 剔除上市不足N年的股票
    stock_listed = df.groupby('stock_id')['date'].agg(['min', 'max'])
    stock_listed['years'] = (stock_listed['max'] - stock_listed['min']).dt.days / 365
    valid_stocks = stock_listed[stock_listed['years'] >= min_listed_years].index.tolist()
    df = df[df['stock_id'].isin(valid_stocks)]

    # 剔除ST/停牌/流动性差的记录
    df = df[(df['close'] > 0) & (df['volume'] > 0)]
    if 'amount' in df.columns:
        df = df[df['amount'] > MIN_DAILY_AMOUNT]

    print(f"预处理后剩余 {len(df)} 条记录，{df['stock_id'].nunique()} 只股票")
    return df


def add_index_features(stock_df, index_df):
    """添加沪深300指数特征"""
    if index_df is None or len(index_df) == 0:
        return stock_df

    index_df = index_df.copy()
    index_df['date'] = pd.to_datetime(index_df['date'])
    index_df = index_df.sort_values('date')

    index_df['index_return'] = index_df['index_close'].pct_change()
    index_df['index_volatility_20'] = index_df['index_return'].rolling(20).std()
    index_df['index_ma5'] = index_df['index_close'].rolling(5).mean()
    index_df['index_ma20'] = index_df['index_close'].rolling(20).mean()

    stock_df = stock_df.merge(
        index_df[['date', 'index_close', 'index_return', 'index_volatility_20', 'index_ma5', 'index_ma20']],
        on='date', how='left'
    )

    return stock_df


def add_event_features(df):
    """添加事件特征"""
    df = df.copy()
    df['is_in_index'] = 1
    months = df['date'].dt.month
    df['earnings_season'] = months.isin([3, 4, 8, 9, 10, 11]).astype(int)
    return df


def _get_baostock_latest_date():
    """快速查询baostock最新交易日期（仅查沪深300指数一条记录）"""
    try:
        import baostock as bs
    except ImportError:
        return None

    try:
        lg = bs.login()
        if lg.error_code != '0':
            return None
        rs = bs.query_history_k_data_plus(
            "sh.000300", "date", end_date=datetime.now().strftime('%Y-%m-%d'),
            start_date=(datetime.now() - pd.Timedelta(days=7)).strftime('%Y-%m-%d'),
            frequency='d'
        )
        latest = None
        while rs.error_code == '0' and rs.next():
            latest = rs.get_row_data()[0]
        bs.logout()
        return latest
    except Exception:
        return None


def _get_local_latest_date(csv_path):
    """读取本地CSV中最新的日期"""
    if not os.path.exists(csv_path):
        return None
    try:
        df = pd.read_csv(csv_path, usecols=['date'])
        return str(df['date'].max())
    except Exception:
        return None


def load_all_data(data_dir=None, force_refresh=True, exclude_last_trading_days=0):
    """加载并融合所有数据源（智能刷新：仅当baostock有新数据时才重新下载）

    参数:
        exclude_last_trading_days: 排除最近N个交易日的数据（用于训练时留出最新数据做预测）
    """
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')

    stock_csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'stock_data.csv')
    index_csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'index_data.csv')

    local_exists = os.path.exists(stock_csv_path) and os.path.exists(index_csv_path)

    if force_refresh and local_exists:
        # 有本地缓存时，先快速比对日期，避免重复下载
        remote_date = _get_baostock_latest_date()
        local_date = _get_local_latest_date(stock_csv_path)

        if remote_date is None:
            # 无网络/baostock不可用，直接用本地数据
            print("baostock不可达（无网络或服务异常），使用本地CSV缓存...")
            stock_df = pd.read_csv(stock_csv_path)
            index_df = pd.read_csv(index_csv_path)
            print(f"成功加载本地股票数据: {len(stock_df)} 条记录，{stock_df['stock_id'].nunique()} 只股票")
            print(f"成功加载本地指数数据: {len(index_df)} 条记录")
        elif local_date is not None and remote_date <= local_date:
            # 远程日期不新于本地，无需下载
            print(f"本地数据已是最新 (本地: {local_date}, 远程: {remote_date})，跳过下载")
            stock_df = pd.read_csv(stock_csv_path)
            index_df = pd.read_csv(index_csv_path)
            print(f"成功加载本地股票数据: {len(stock_df)} 条记录，{stock_df['stock_id'].nunique()} 只股票")
            print(f"成功加载本地指数数据: {len(index_df)} 条记录")
        else:
            # 远程有新数据，重新下载
            print(f"远程有新数据 (本地: {local_date}, 远程: {remote_date})，重新下载...")
            stock_df, index_df = download_baostock_data()
            if stock_df is None or len(stock_df) == 0:
                print("baostock下载失败，降级到本地CSV缓存...")
                stock_df = pd.read_csv(stock_csv_path)
                index_df = pd.read_csv(index_csv_path)
                print(f"成功加载本地股票数据: {len(stock_df)} 条记录，{stock_df['stock_id'].nunique()} 只股票")
                print(f"成功加载本地指数数据: {len(index_df)} 条记录")
    elif force_refresh and not local_exists:
        # 无本地缓存，直接下载
        print(f"无本地缓存，动态{LOOKBACK_DAYS}天窗口，从baostock下载最新数据...")
        stock_df, index_df = download_baostock_data()
        if stock_df is None or len(stock_df) == 0:
            print("错误：无法获取数据，baostock下载失败且无本地缓存")
            raise RuntimeError("无法获取数据，请确保网络连接正常或提供本地数据")
    elif local_exists:
        print("从本地CSV文件加载数据...")
        stock_df = pd.read_csv(stock_csv_path)
        index_df = pd.read_csv(index_csv_path)
        print(f"成功加载本地股票数据: {len(stock_df)} 条记录，{stock_df['stock_id'].nunique()} 只股票")
        print(f"成功加载本地指数数据: {len(index_df)} 条记录")
    else:
        print(f"动态{LOOKBACK_DAYS}天窗口，从baostock下载最新数据...")
        stock_df, index_df = download_baostock_data()
        if stock_df is None or len(stock_df) == 0:
            print("错误：无法获取数据")
            raise RuntimeError("无法获取数据，请确保网络连接正常或提供本地数据")

    # 宏观数据：尝试AKShare真实数据，失败则回退到模拟数据
    macro_df = download_akshare_macro_data()
    if macro_df is None:
        macro_df = generate_macro_data()
    # 行业数据：目前用模拟数据（AKShare申万行业指数接口不稳定）
    industry_df = generate_industry_data()

    # 预处理
    if stock_df is not None and len(stock_df) > 0:
        stock_df = preprocess_stock_data(stock_df)
        stock_df = add_index_features(stock_df, index_df)
        stock_df = add_event_features(stock_df)

    # 排除最近N个交易日（用于训练时留出最新数据做预测）
    if exclude_last_trading_days > 0 and stock_df is not None and len(stock_df) > 0:
        all_dates = sorted(stock_df['date'].unique())
        if len(all_dates) > exclude_last_trading_days:
            excluded_dates = set(all_dates[-exclude_last_trading_days:])
            stock_df = stock_df[~stock_df['date'].isin(excluded_dates)]
            print(f"已排除最近 {exclude_last_trading_days} 个交易日: {sorted(excluded_dates)}")
            print(f"排除后剩余 {len(stock_df)} 条记录，{stock_df['stock_id'].nunique()} 只股票")

    return stock_df, industry_df, macro_df


def generate_simulated_stock_data(n_stocks=300):
    """生成模拟股票数据（离线备用模式）"""
    print(f"生成 {n_stocks} 只模拟股票数据...")
    np.random.seed(42)
    start_date = (datetime.now() - pd.Timedelta(days=LOOKBACK_DAYS)).strftime('%Y-%m-%d')
    dates = pd.date_range(start_date, datetime.now().strftime('%Y-%m-%d'), freq='B')

    stocks = [f'{i:06d}' for i in range(1, n_stocks + 1)]
    data_list = []

    for stock_id in stocks:
        price = np.random.uniform(5, 100)
        prices = price * np.cumprod(1 + np.random.randn(len(dates)) * 0.02 + 0.0003)

        df_stock = pd.DataFrame({
            'stock_id': stock_id,
            'date': dates,
            'open': prices * (1 + np.random.randn(len(dates)) * 0.005),
            'high': prices * (1 + np.abs(np.random.randn(len(dates)) * 0.01)),
            'low': prices * (1 - np.abs(np.random.randn(len(dates)) * 0.01)),
            'close': prices,
            'volume': np.random.randint(5000000, 50000000, len(dates)),
            'amount': np.random.randint(50000000, 5000000000, len(dates)),
            'turnover_rate': np.random.rand(len(dates)) * 5 + 0.5,
            'pe': np.random.rand(len(dates)) * 30 + 10,
            'pb': np.random.rand(len(dates)) * 3 + 0.5,
        })
        data_list.append(df_stock)

    return pd.concat(data_list, ignore_index=True)


if __name__ == "__main__":
    stock_df, industry_df, macro_df = load_all_data()
    print(f"股票数据: {len(stock_df) if stock_df is not None else 0} 条")
    print(f"行业数据: {len(industry_df) if industry_df is not None else 0} 条")
    print(f"宏观数据: {len(macro_df) if macro_df is not None else 0} 条")
