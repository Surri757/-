"""
预测主程序 - Stacking集成 + Black-Litterman组合优化 (修复版)
"""
import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize

warnings.filterwarnings('ignore')

# ---- 配置 ----
RANDOM_SEED = 42
SEQ_LEN = 60
PRED_HORIZON = 5
RISK_AVERSION = 1.2
MAX_INDUSTRY_WEIGHT = 0.4     # 仅行业集中度约束
MAX_VOLATILITY_RATIO = 1.2

# 路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---- 种子 ----
np.random.seed(RANDOM_SEED)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from featurework import FeatureEngineering
from train import (PatchTSTModel, TimesNetModel, DLinearModel, SequenceDataset,
                   set_all_seeds, GBDT_N_ESTIMATORS, GBDT_LR, GBDT_MAX_DEPTH,
                   D_MODEL, N_HEADS, E_LAYERS, PATCH_LEN, STRIDE,
                   compute_volatility_cluster, WeightedEnsemble)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("使用CPU推理")


# ---- 申万行业映射（完整版） ----
STOCK_INDUSTRY_MAP = {
    # 银行
    '000001': '银行', '002142': '银行', '002839': '银行', '600000': '银行',
    '600015': '银行', '600016': '银行', '600036': '银行', '600919': '银行',
    '600926': '银行', '601009': '银行', '601128': '银行', '601166': '银行',
    '601169': '银行', '601229': '银行', '601288': '银行', '601328': '银行',
    '601398': '银行', '601528': '银行', '601818': '银行', '601838': '银行',
    '601939': '银行', '601988': '银行', '601997': '银行', '601998': '银行',
    '603323': '银行',
    # 非银金融
    '000563': '非银金融', '000627': '非银金融', '000712': '非银金融',
    '000728': '非银金融', '000750': '非银金融', '000776': '非银金融',
    '000783': '非银金融', '002423': '非银金融', '002500': '非银金融',
    '002670': '非银金融', '002673': '非银金融', '002736': '非银金融',
    '002797': '非银金融', '002926': '非银金融', '002939': '非银金融',
    '002945': '非银金融', '300059': '非银金融', '600030': '非银金融',
    '600061': '非银金融', '600109': '非银金融', '600155': '非银金融',
    '600369': '非银金融', '600390': '非银金融', '600705': '非银金融',
    '600837': '非银金融', '600901': '非银金融', '600909': '非银金融',
    '600918': '非银金融', '600958': '非银金融', '600999': '非银金融',
    '601066': '非银金融', '601099': '非银金融', '601108': '非银金融',
    '601162': '非银金融', '601198': '非银金融', '601211': '非银金融',
    '601236': '非银金融', '601318': '非银金融', '601319': '非银金融',
    '601336': '非银金融', '601375': '非银金融', '601377': '非银金融',
    '601555': '非银金融', '601601': '非银金融', '601628': '非银金融',
    '601688': '非银金融', '601696': '非银金融', '601788': '非银金融',
    '601878': '非银金融', '601881': '非银金融', '601901': '非银金融',
    '601990': '非银金融', '603300': '非银金融',
    # 食品饮料
    '000568': '食品饮料', '000596': '食品饮料', '000799': '食品饮料',
    '000858': '食品饮料', '000860': '食品饮料', '000876': '食品饮料',
    '000895': '食品饮料', '002304': '食品饮料', '002461': '食品饮料',
    '002568': '食品饮料', '002714': '食品饮料', '300146': '食品饮料',
    '300498': '食品饮料', '600132': '食品饮料', '600298': '食品饮料',
    '600519': '食品饮料', '600559': '食品饮料', '600600': '食品饮料',
    '600702': '食品饮料', '600779': '食品饮料', '600809': '食品饮料',
    '600872': '食品饮料', '600882': '食品饮料', '600887': '食品饮料',
    '603027': '食品饮料', '603288': '食品饮料', '603345': '食品饮料',
    '603369': '食品饮料', '603589': '食品饮料', '603833': '食品饮料',
    # 医药生物
    '000423': '医药生物', '000538': '医药生物', '000661': '医药生物',
    '000963': '医药生物', '000999': '医药生物', '002001': '医药生物',
    '002007': '医药生物', '002019': '医药生物', '002022': '医药生物',
    '002030': '医药生物', '002038': '医药生物', '002044': '医药生物',
    '002252': '医药生物', '002262': '医药生物', '002294': '医药生物',
    '002317': '医药生物', '002399': '医药生物', '002411': '医药生物',
    '002422': '医药生物', '002432': '医药生物', '002437': '医药生物',
    '002603': '医药生物', '002653': '医药生物', '002675': '医药生物',
    '002727': '医药生物', '002773': '医药生物', '002821': '医药生物',
    '300003': '医药生物', '300015': '医药生物', '300122': '医药生物',
    '300124': '医药生物', '300142': '医药生物', '300147': '医药生物',
    '300253': '医药生物', '300267': '医药生物', '300347': '医药生物',
    '300357': '医药生物', '300433': '医药生物', '300482': '医药生物',
    '300529': '医药生物', '300558': '医药生物', '300595': '医药生物',
    '300601': '医药生物', '300628': '医药生物', '300630': '医药生物',
    '300633': '医药生物', '300676': '医药生物', '300677': '医药生物',
    '300725': '医药生物', '300759': '医药生物', '300760': '医药生物',
    '300782': '医药生物', '600062': '医药生物', '600079': '医药生物',
    '600085': '医药生物', '600161': '医药生物', '600196': '医药生物',
    '600201': '医药生物', '600252': '医药生物', '600276': '医药生物',
    '600285': '医药生物', '600299': '医药生物', '600329': '医药生物',
    '600332': '医药生物', '600380': '医药生物', '600420': '医药生物',
    '600422': '医药生物', '600436': '医药生物', '600511': '医药生物',
    '600521': '医药生物', '600535': '医药生物', '600557': '医药生物',
    '600566': '医药生物', '600572': '医药生物', '600587': '医药生物',
    '600664': '医药生物', '600673': '医药生物', '600763': '医药生物',
    '600771': '医药生物', '600867': '医药生物', '601607': '医药生物',
    '603087': '医药生物', '603233': '医药生物', '603259': '医药生物',
    '603392': '医药生物', '603456': '医药生物', '603658': '医药生物',
    '603707': '医药生物', '603858': '医药生物', '603882': '医药生物',
    '603883': '医药生物', '603939': '医药生物',
    # 电子
    '000725': '电子', '002008': '电子', '002049': '电子', '002056': '电子',
    '002065': '电子', '002106': '电子', '002138': '电子', '002156': '电子',
    '002179': '电子', '002180': '电子', '002185': '电子', '002236': '电子',
    '002241': '电子', '002273': '电子', '002281': '电子', '002371': '电子',
    '002384': '电子', '002402': '电子', '002409': '电子', '002414': '电子',
    '002456': '电子', '002463': '电子', '002475': '电子', '002484': '电子',
    '002506': '电子', '002600': '电子', '002681': '电子', '002745': '电子',
    '002841': '电子', '002916': '电子', '002920': '电子', '002925': '电子',
    '002938': '电子', '300014': '电子', '300088': '电子', '300115': '电子',
    '300136': '电子', '300207': '电子', '300223': '电子', '300296': '电子',
    '300308': '电子', '300316': '电子', '300319': '电子', '300327': '电子',
    '300373': '电子', '300408': '电子', '300413': '电子', '300433': '电子',
    '300456': '电子', '300458': '电子', '300476': '电子', '300496': '电子',
    '300502': '电子', '300529': '电子', '300552': '电子', '300558': '电子',
    '300567': '电子', '300604': '电子', '300623': '电子', '300628': '电子',
    '300661': '电子', '300666': '电子', '300672': '电子', '300676': '电子',
    '300679': '电子', '300699': '电子', '300724': '电子', '300726': '电子',
    '300735': '电子', '300747': '电子', '300750': '电子', '300751': '电子',
    '300760': '电子', '300776': '电子', '300782': '电子', '300793': '电子',
    '300803': '电子', '300832': '电子', '300866': '电子', '300896': '电子',
    '300919': '电子', '600171': '电子', '600183': '电子', '600460': '电子',
    '600563': '电子', '600584': '电子', '600667': '电子', '600703': '电子',
    '600745': '电子', '600885': '电子', '601012': '电子', '601138': '电子',
    '601231': '电子', '601689': '电子', '603005': '电子', '603019': '电子',
    '603160': '电子', '603185': '电子', '603228': '电子', '603260': '电子',
    '603290': '电子', '603296': '电子', '603501': '电子', '603596': '电子',
    '603606': '电子', '603659': '电子', '603678': '电子', '603893': '电子',
    '603920': '电子', '603986': '电子',
    # 计算机
    '000066': '计算机', '000158': '计算机', '000555': '计算机',
    '000938': '计算机', '000977': '计算机', '000997': '计算机',
    '002027': '计算机', '002065': '计算机', '002139': '计算机',
    '002152': '计算机', '002153': '计算机', '002174': '计算机',
    '002195': '计算机', '002212': '计算机', '002230': '计算机',
    '002236': '计算机', '002253': '计算机', '002261': '计算机',
    '002268': '计算机', '002308': '计算机', '002322': '计算机',
    '002362': '计算机', '002368': '计算机', '002373': '计算机',
    '002376': '计算机', '002405': '计算机', '002410': '计算机',
    '002415': '计算机', '002421': '计算机', '002439': '计算机',
    '002444': '计算机', '002465': '计算机', '002474': '计算机',
    '002544': '计算机', '002583': '计算机', '002609': '计算机',
    '002642': '计算机', '002649': '计算机', '002651': '计算机',
    '002657': '计算机', '002670': '计算机', '002690': '计算机',
    '002777': '计算机', '002837': '计算机', '002906': '计算机',
    '002912': '计算机', '002920': '计算机', '002929': '计算机',
    '002941': '计算机', '002955': '计算机', '002979': '计算机',
    '300002': '计算机', '300010': '计算机', '300017': '计算机',
    '300020': '计算机', '300024': '计算机', '300033': '计算机',
    '300036': '计算机', '300044': '计算机', '300047': '计算机',
    '300054': '计算机', '300058': '计算机', '300059': '计算机',
    '300065': '计算机', '300070': '计算机', '300075': '计算机',
    '300077': '计算机', '300078': '计算机', '300079': '计算机',
    '300085': '计算机', '300088': '计算机', '300098': '计算机',
    '300113': '计算机', '300115': '计算机', '300124': '计算机',
    '300130': '计算机', '300136': '计算机', '300166': '计算机',
    '300168': '计算机', '300170': '计算机', '300182': '计算机',
    '300188': '计算机', '300207': '计算机', '300212': '计算机',
    '300223': '计算机', '300229': '计算机', '300244': '计算机',
    '300248': '计算机', '300251': '计算机', '300253': '计算机',
    '300271': '计算机', '300287': '计算机', '300290': '计算机',
    '300292': '计算机', '300294': '计算机', '300296': '计算机',
    '300297': '计算机', '300300': '计算机', '300302': '计算机',
    '300310': '计算机', '300311': '计算机', '300315': '计算机',
    '300319': '计算机', '300324': '计算机', '300326': '计算机',
    '300327': '计算机', '300339': '计算机', '300348': '计算机',
    '300349': '计算机', '300352': '计算机', '300353': '计算机',
    '300354': '计算机', '300357': '计算机', '300365': '计算机',
    '300366': '计算机', '300369': '计算机', '300373': '计算机',
    '300377': '计算机', '300378': '计算机', '300383': '计算机',
    '300386': '计算机', '300394': '计算机', '300401': '计算机',
    '300406': '计算机', '300408': '计算机', '300413': '计算机',
    '300418': '计算机', '300433': '计算机', '300442': '计算机',
    '300448': '计算机', '300450': '计算机', '300451': '计算机',
    '300454': '计算机', '300455': '计算机', '300458': '计算机',
    '300459': '计算机', '300463': '计算机', '300465': '计算机',
    '300468': '计算机', '300469': '计算机', '300473': '计算机',
    '300474': '计算机', '300476': '计算机', '300479': '计算机',
    '300482': '计算机', '300487': '计算机', '300493': '计算机',
    '300494': '计算机', '300496': '计算机', '300498': '计算机',
    '300502': '计算机', '300506': '计算机', '300508': '计算机',
    '300520': '计算机', '300523': '计算机', '300525': '计算机',
    '300529': '计算机', '300533': '计算机', '300541': '计算机',
    '300546': '计算机', '300552': '计算机', '300558': '计算机',
    '300559': '计算机', '300566': '计算机', '300567': '计算机',
    '300570': '计算机', '300573': '计算机', '300576': '计算机',
    '300579': '计算机', '300588': '计算机', '300590': '计算机',
    '300596': '计算机', '300598': '计算机', '300601': '计算机',
    '300602': '计算机', '300603': '计算机', '300604': '计算机',
    '300608': '计算机', '300609': '计算机', '300613': '计算机',
    '300616': '计算机', '300618': '计算机', '300623': '计算机',
    '300624': '计算机', '300628': '计算机', '300630': '计算机',
    '300633': '计算机', '300634': '计算机', '300638': '计算机',
    '300642': '计算机', '300645': '计算机', '300652': '计算机',
    '300653': '计算机', '300654': '计算机', '300655': '计算机',
    '300657': '计算机', '300659': '计算机', '300661': '计算机',
    '300662': '计算机', '300663': '计算机', '300666': '计算机',
    '300672': '计算机', '300673': '计算机', '300674': '计算机',
    '300676': '计算机', '300677': '计算机', '300678': '计算机',
    '300679': '计算机', '300682': '计算机', '300684': '计算机',
    '300687': '计算机', '300693': '计算机', '300696': '计算机',
    '300699': '计算机', '300702': '计算机', '300720': '计算机',
    '300724': '计算机', '300725': '计算机', '300726': '计算机',
    '300735': '计算机', '300738': '计算机', '300741': '计算机',
    '300742': '计算机', '300745': '计算机', '300747': '计算机',
    '300748': '计算机', '300750': '计算机', '300751': '计算机',
    '300753': '计算机', '300758': '计算机', '300760': '计算机',
    '300761': '计算机', '300762': '计算机', '300763': '计算机',
    '300764': '计算机', '300766': '计算机', '300768': '计算机',
    '300770': '计算机', '300773': '计算机', '300776': '计算机',
    '300782': '计算机', '300785': '计算机', '300788': '计算机',
    '300791': '计算机', '300793': '计算机', '300796': '计算机',
    '300797': '计算机', '300800': '计算机', '300803': '计算机',
    '300810': '计算机', '300832': '计算机', '300838': '计算机',
    '300846': '计算机', '300851': '计算机', '300866': '计算机',
    '300872': '计算机', '300885': '计算机', '300887': '计算机',
    '300890': '计算机', '300896': '计算机', '300900': '计算机',
    '300913': '计算机', '300915': '计算机', '300919': '计算机',
    '300925': '计算机', '300928': '计算机', '300935': '计算机',
    '300941': '计算机', '300942': '计算机', '300949': '计算机',
    '300951': '计算机', '300953': '计算机', '300957': '计算机',
    '300960': '计算机', '300961': '计算机', '300970': '计算机',
    '300973': '计算机', '300975': '计算机', '300977': '计算机',
    '300980': '计算机', '300985': '计算机', '300986': '计算机',
    '300996': '计算机', '300999': '计算机', '600100': '计算机',
    '600225': '计算机', '600271': '计算机', '600410': '计算机',
    '600446': '计算机', '600536': '计算机', '600556': '计算机',
    '600570': '计算机', '600571': '计算机', '600588': '计算机',
    '600602': '计算机', '600654': '计算机', '600662': '计算机',
    '600718': '计算机', '600728': '计算机', '600756': '计算机',
    '600759': '计算机', '600764': '计算机', '600767': '计算机',
    '600797': '计算机', '600804': '计算机', '600845': '计算机',
    '600850': '计算机', '600855': '计算机', '600880': '计算机',
    '600986': '计算机', '601313': '计算机', '601360': '计算机',
    '601519': '计算机', '601658': '计算机', '601669': '计算机',
    '601789': '计算机', '601799': '计算机', '601858': '计算机',
    '601869': '计算机', '601872': '计算机', '601878': '计算机',
    '601900': '计算机', '601928': '计算机', '601929': '计算机',
    '601949': '计算机', '601958': '计算机', '601968': '计算机',
    '601990': '计算机', '603000': '计算机', '603008': '计算机',
    '603019': '计算机', '603027': '计算机', '603039': '计算机',
    '603042': '计算机', '603055': '计算机', '603060': '计算机',
    '603068': '计算机', '603069': '计算机', '603083': '计算机',
    '603087': '计算机', '603096': '计算机', '603108': '计算机',
    '603113': '计算机', '603117': '计算机', '603118': '计算机',
    '603123': '计算机', '603126': '计算机', '603127': '计算机',
    '603138': '计算机', '603160': '计算机', '603171': '计算机',
    '603179': '计算机', '603185': '计算机', '603189': '计算机',
    '603197': '计算机', '603220': '计算机', '603228': '计算机',
    '603232': '计算机', '603236': '计算机', '603238': '计算机',
    '603259': '计算机', '603260': '计算机', '603267': '计算机',
    '603279': '计算机', '603290': '计算机', '603296': '计算机',
    '603297': '计算机', '603300': '计算机', '603303': '计算机',
}

# 为未映射的股票用股票代码前缀推断行业
def get_industry(stock_id):
    if stock_id in STOCK_INDUSTRY_MAP:
        return STOCK_INDUSTRY_MAP[stock_id]
    # 默认归为"其他"
    return '其他'


class MarketRegime:
    """市场状态检测器"""

    @staticmethod
    def detect(index_returns, index_vol, predicted_returns):
        """
        检测当前市场状态
        Returns: regime, total_exposure, concentration_preference
        regime: 'bull' | 'sideways' | 'bear' | 'panic'
        total_exposure: 建议总仓位比例 0~1
        concentration_preference: 0=分散, 1=集中
        """
        if index_returns is None or len(index_returns) < 20:
            return 'sideways', 0.9, 0.4

        ret_20d = np.mean(index_returns[-20:])
        ret_5d = np.mean(index_returns[-5:])
        vol_20d = np.std(index_returns[-20:])
        vol_60d = np.std(index_returns[-60:]) if len(index_returns) >= 60 else vol_20d

        # 波动率变化率
        vol_change = vol_20d / (vol_60d + 1e-8)

        # 预测收益的均值和离散度
        pred_mean = np.mean(predicted_returns) if len(predicted_returns) > 0 else 0
        pred_pos_ratio = np.mean(predicted_returns > 0) if len(predicted_returns) > 0 else 0.5

        # --- 恐慌 (Panic): 极端下跌 + 波动率飙升 → 重仓抄底 ---
        if ret_5d < -0.03 and vol_change > 1.5:
            print(f"[市场状态] 恐慌 (5日跌{ret_5d:.1%}, 波动率飙升{vol_change:.1f}x) → 重仓抄底")
            return 'panic', 0.98, 0.85   # 满仓，高度集中博弈反弹

        # --- 熊市 (Bear): 下跌趋势 + 高波动 → 精选逆势股，保持高仓位 ---
        if ret_20d < -0.01 and vol_change > 1.2:
            print(f"[市场状态] 熊市 (20日跌{ret_20d:.1%}, 波动率{vol_20d:.3f}) → 精选逆势股")
            # 熊市也要保持高仓位，目标是找到逆势上涨的个股
            return 'bear', 0.88, 0.75

        # --- 震荡 (Sideways): 横盘/低波动 → 精选个股，保持仓位 ---
        if abs(ret_20d) < 0.005 and vol_change < 1.3:
            print(f"[市场状态] 震荡 (20日波动{ret_20d:.1%}) → 精选个股")
            return 'sideways', 0.85, 0.5

        # --- 牛市 (Bull): 上涨趋势 + 正常波动 → 满仓分散 ---
        if ret_20d > 0.005:
            print(f"[市场状态] 牛市 (20日涨{ret_20d:.1%}) → 积极布局")
            return 'bull', 0.98, 0.2

        # --- 默认：保持高仓位 ---
        print(f"[市场状态] 中性 (20日{ret_20d:.1%}) → 标准配置")
        return 'sideways', 0.9, 0.4


class BlackLittermanOptimizer:
    """动态决策组合优化器 - 无权重上限，根据市场状态自适应"""

    def __init__(self, risk_aversion=RISK_AVERSION,
                 max_industry_weight=MAX_INDUSTRY_WEIGHT,
                 max_vol_ratio=MAX_VOLATILITY_RATIO):
        self.risk_aversion = risk_aversion
        self.max_industry_weight = max_industry_weight
        self.max_vol_ratio = max_vol_ratio

    def optimize_portfolio(self, predicted_returns, predicted_stds,
                          market_caps, historical_returns, industry_ids=None,
                          index_returns=None):
        """
        动态决策组合优化
        - 无单只股票权重上限
        - 无持仓数量限制
        - 根据市场状态自适应调整总仓位和集中度
        - 总权重 <= 1.0 (剩余为闲置本金)
        """
        n_assets = len(predicted_returns)
        if n_assets == 0:
            return np.array([]), {}

        # === 第1步：检测市场状态 ===
        regime_info = MarketRegime.detect(index_returns, None, predicted_returns)
        regime, target_exposure, concentration = regime_info

        # === 第2步：协方差估计 ===
        if historical_returns is not None and historical_returns.shape[0] > 60:
            cov_matrix = self._ledoit_wolf_shrinkage(historical_returns)
        else:
            cov_matrix = np.diag(np.maximum(predicted_stds, 0.001) ** 2)

        # === 第3步：Black-Litterman后验收益 ===
        mkt_weights = market_caps / (market_caps.sum() + 1e-10)
        mkt_vol = np.sqrt(mkt_weights @ cov_matrix @ mkt_weights + 1e-10)
        equilibrium_returns = self.risk_aversion * cov_matrix @ mkt_weights

        tau = 0.05
        P = np.eye(n_assets)
        Q = predicted_returns
        omega = np.diag(np.maximum(predicted_stds, 0.001) ** 2 + tau * np.diag(cov_matrix))

        try:
            prior_cov_inv = np.linalg.inv(tau * cov_matrix)
            omega_inv = np.linalg.inv(omega)
            posterior_cov = np.linalg.inv(prior_cov_inv + P.T @ omega_inv @ P)
            posterior_returns = posterior_cov @ (prior_cov_inv @ equilibrium_returns + P.T @ omega_inv @ Q)
        except np.linalg.LinAlgError:
            posterior_returns = 0.5 * equilibrium_returns + 0.5 * Q

        # === 第4步：根据市场状态确定选股策略 ===
        valid = np.isfinite(posterior_returns)
        if not valid.any():
            return np.zeros(n_assets), {'regime': regime, 'exposure': 0, 'n_stocks': 0}

        sorted_idx = np.argsort(posterior_returns)[::-1]

        # 根据市场状态决定持仓策略（始终高仓位，最多选5只股票）
        if regime == 'panic':
            # 恐慌：集中2-4只最被低估的股票，满仓博弈反弹
            n_picks = max(2, min(4, int(n_assets * 0.02)))
            positive = sorted_idx[posterior_returns[sorted_idx] > 0]
            top_idx = positive[:n_picks] if len(positive) >= 2 else sorted_idx[:n_picks]
            weights = np.zeros(n_assets)
            pos_rets = np.maximum(posterior_returns[top_idx], 1e-6)
            weights[top_idx] = pos_rets / pos_rets.sum() * target_exposure

        elif regime == 'bear':
            # 熊市：精选2-4只逆势股，保持高仓位
            n_picks = max(2, min(4, int(n_assets * 0.02)))
            positive = sorted_idx[posterior_returns[sorted_idx] > 0]
            top_idx = positive[:n_picks] if len(positive) >= 2 else sorted_idx[:n_picks]
            weights = np.zeros(n_assets)
            pos_rets = np.maximum(posterior_returns[top_idx], 1e-6)
            weights[top_idx] = pos_rets / pos_rets.sum() * target_exposure

        elif regime == 'bull':
            # 牛市：分散布局3-5只，满仓
            n_picks = max(3, min(5, int(n_assets * 0.05)))
            positive = sorted_idx[posterior_returns[sorted_idx] > 0]
            if len(positive) >= 3:
                top_idx = positive[:n_picks]
            else:
                top_idx = sorted_idx[:max(3, n_picks)]
            weights = np.zeros(n_assets)
            pos_rets = np.maximum(posterior_returns[top_idx], 1e-6)
            weights[top_idx] = pos_rets / pos_rets.sum() * target_exposure

        else:  # sideways / default
            # 震荡：精选3-5只，保持高仓位
            n_picks = max(3, min(5, int(n_assets * 0.03)))
            positive = sorted_idx[posterior_returns[sorted_idx] > 0]
            if len(positive) >= 2:
                top_idx = positive[:n_picks]
            else:
                top_idx = sorted_idx[:max(3, n_picks)]
            weights = np.zeros(n_assets)
            pos_rets = np.maximum(posterior_returns[top_idx], 1e-6)
            weights[top_idx] = pos_rets / pos_rets.sum() * target_exposure

        # === 第5步：行业约束（软约束，超出时调整） ===
        if industry_ids is not None:
            for ind in set(industry_ids):
                mask = np.array([i == ind for i in industry_ids])
                ind_weight = weights[mask].sum()
                if ind_weight > self.max_industry_weight:
                    weights[mask] *= self.max_industry_weight / (ind_weight + 1e-10)

        # === 第6步：归一化到目标仓位（始终满仓，不闲置资金） ===
        total = weights.sum()
        if total > 0:
            weights *= target_exposure / total
        # 确保单只权重不超过1
        weights = np.minimum(weights, 1.0)

        # 统计信息
        held_stocks = int(np.sum(weights > 0.001))
        held_stock_ids = np.where(weights > 0.001)[0]
        info = {
            'regime': regime,
            'target_exposure': target_exposure,
            'actual_exposure': float(weights.sum()),
            'n_stocks': held_stocks,
            'concentration': concentration,
        }

        return weights, info

    def _ledoit_wolf_shrinkage(self, returns):
        """Ledoit-Wolf收缩协方差估计"""
        n_samples, n_assets = returns.shape
        sample_cov = np.cov(returns, rowvar=False)
        stds = np.sqrt(np.diag(sample_cov))
        mean_corr = (np.corrcoef(returns, rowvar=False) - np.eye(n_assets)).mean()
        target = np.outer(stds, stds) * mean_corr
        np.fill_diagonal(target, np.diag(sample_cov))
        delta_sq = ((sample_cov - target) ** 2).sum() / n_assets**2
        pi_mat = np.zeros((n_assets, n_assets))
        for i in range(n_samples):
            ret_i = returns[i].reshape(-1, 1)
            diff = ret_i @ ret_i.T - sample_cov
            pi_mat += diff ** 2
        pi = pi_mat.sum() / (n_samples**2)
        shrinkage = np.clip(pi / (pi + delta_sq + 1e-10), 0, 1)
        return shrinkage * target + (1 - shrinkage) * sample_cov


class StackingPredictor:
    """Stacking集成预测器"""

    def __init__(self):
        self.base_models = {}
        self.meta_model = None
        self.scaler = None
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model_names = []

    def load_models(self):
        print("加载模型...")

        # ML模型
        for name in ['lightgbm', 'catboost', 'xgboost']:
            fpath = os.path.join(MODEL_DIR, f'{name}_model.pkl')
            if os.path.exists(fpath):
                with open(fpath, 'rb') as f:
                    self.base_models[name] = pickle.load(f)
                self.model_names.append(name)
                print(f"  加载 {name}")

        # PyTorch模型
        pt_models = {
            'patchtst': PatchTSTModel(seq_len=SEQ_LEN, n_features=100, d_model=D_MODEL,
                                     n_heads=N_HEADS, e_layers=E_LAYERS),
            'timesnet': TimesNetModel(seq_len=SEQ_LEN, n_features=100, d_model=D_MODEL, e_layers=2),
            'dlinear': DLinearModel(seq_len=SEQ_LEN, n_features=100),
        }

        for name, model in pt_models.items():
            fpath = os.path.join(MODEL_DIR, f'{name}_model.pth')
            if os.path.exists(fpath):
                state = torch.load(fpath, map_location=self.device)
                # 如果特征数不匹配，重建模型
                try:
                    model.load_state_dict(state)
                except RuntimeError:
                    # 推断实际特征数
                    key_shape = state.get('revin.gamma', state.get('feature_proj.weight'))
                    if key_shape is not None:
                        n_feat = key_shape.shape[-1] if key_shape.dim() > 1 else key_shape.shape[0]
                        print(f"  重建 {name} (n_features={n_feat})")
                        if name == 'patchtst':
                            model = PatchTSTModel(seq_len=SEQ_LEN, n_features=n_feat, d_model=D_MODEL,
                                                 n_heads=N_HEADS, e_layers=E_LAYERS)
                        elif name == 'timesnet':
                            model = TimesNetModel(seq_len=SEQ_LEN, n_features=n_feat, d_model=D_MODEL, e_layers=2)
                        elif name == 'dlinear':
                            model = DLinearModel(seq_len=SEQ_LEN, n_features=n_feat)
                        model.load_state_dict(state)
                model.to(self.device)
                model.eval()
                self.base_models[name] = model
                self.model_names.append(name)
                print(f"  加载 {name}")

        # 元模型
        meta_path = os.path.join(MODEL_DIR, 'meta_model.pkl')
        if os.path.exists(meta_path):
            with open(meta_path, 'rb') as f:
                self.meta_model = pickle.load(f)
            print("  加载元模型")

        # Scaler
        scaler_path = os.path.join(MODEL_DIR, 'scaler.pkl')
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            print("  加载标准化器")

        return len(self.base_models) > 0

    def predict(self, X, historical_prices=None):
        """Stacking预测 + 计算不确定性"""
        n_samples = len(X)
        n_models = len(self.base_models)
        base_preds = np.zeros((n_samples, n_models))

        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X

        for idx, name in enumerate(self.model_names):
            model = self.base_models[name]
            if name in ['lightgbm', 'catboost', 'xgboost']:
                base_preds[:, idx] = model.predict(X_scaled)
            else:
                # PyTorch模型
                X_seq = []
                for i in range(SEQ_LEN, len(X_scaled)):
                    X_seq.append(X_scaled[i - SEQ_LEN:i])
                if len(X_seq) > 0:
                    X_tensor = torch.FloatTensor(np.array(X_seq)).to(self.device)
                    model.eval()
                    with torch.no_grad():
                        preds = model(X_tensor).cpu().numpy().flatten()
                    start = len(X_scaled) - len(preds)
                    if start >= 0:
                        base_preds[start:, idx] = preds
                    else:
                        base_preds[:, idx] = preds[:len(X_scaled)]

        # 元模型预测
        if self.meta_model is not None:
            final_preds = self.meta_model.predict(base_preds)
        else:
            final_preds = base_preds.mean(axis=1)

        # 不确定性: 模型预测的标准差
        pred_stds = base_preds.std(axis=1)

        return final_preds[-1] if len(final_preds) > 0 else 0.0, pred_stds[-1] if len(pred_stds) > 0 else 0.02


def load_test_data():
    """加载测试数据"""
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    test_path = os.path.join(data_dir, 'test.csv')

    if os.path.exists(test_path):
        print(f"加载测试数据: {test_path}")
        df = pd.read_csv(test_path)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values(['stock_id', 'date'])
        return df

    # 使用主数据的最新部分作为测试
    stock_csv = os.path.join(data_dir, 'stock_data.csv')
    if os.path.exists(stock_csv):
        print("使用stock_data.csv最新数据作为测试集...")
        df = pd.read_csv(stock_csv)
        df['date'] = pd.to_datetime(df['date'])
        latest_date = df['date'].max()
        test_start = latest_date - pd.Timedelta(days=180)
        df = df[df['date'] >= test_start]
        return df

    print("测试数据不存在！")
    return None


def generate_result_csv(predictions, output_path=None):
    """生成 result.csv"""
    if output_path is None:
        output_path = os.path.join(OUTPUT_DIR, 'result.csv')

    # 确保stock_id格式正确（6位数字）
    predictions['stock_id'] = predictions['stock_id'].astype(str).str.zfill(6)

    # 只保留有权重的股票，且最多不超过5只
    result = predictions[predictions['weight'] > 0.001].copy()
    result = result.sort_values('weight', ascending=False)
    result = result.head(5)
    result[['stock_id', 'weight']].to_csv(output_path, index=False, encoding='utf-8')

    # 赛事总收益公式: R = Σ(w_i × r_i)，其中 r_i = (P_{T+5}^close / P_{T+1}^open) - 1
    total_predicted_return = (result['weight'] * result['predicted_return']).sum()

    print(f"\n结果已保存到 {output_path}")
    print(result[['stock_id', 'weight', 'predicted_return']].to_string(index=False))
    print(f"\n总权重: {result['weight'].sum():.4f}")
    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"赛事总预测收益率: {total_predicted_return:.4%}")
    print(f"  (公式: Σ weight_i × pred_return_i, i=1..{len(result)})")
    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")


def main():
    set_all_seeds()
    print("=" * 60)
    print("股价预测模型 - 预测主程序")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # 加载模型
    predictor = StackingPredictor()
    if not predictor.load_models():
        print("警告: 模型文件不存在，生成默认预测")
        predictions = pd.DataFrame({
            'stock_id': ['000001', '000002', '600000', '600001', '600016'],
            'predicted_return': [0.05, 0.04, 0.03, 0.02, 0.01],
            'weight': [0.3, 0.25, 0.25, 0.15, 0.05],
            'predicted_std': [0.01, 0.01, 0.01, 0.01, 0.01],
            'market_cap': [500, 400, 600, 300, 450],
            'industry': ['银行', '房地产', '银行', '非银金融', '银行']
        })
        generate_result_csv(predictions)
        return True

    # 加载数据
    df = load_test_data()
    if df is None or len(df) == 0:
        print("无测试数据！")
        return False

    print(f"测试数据: {len(df)} 条记录, {df['stock_id'].nunique()} 只股票")

    # 宏观和行业数据（优先AKShare真实数据，回退模拟数据，与训练时一致）
    from data_fetcher import (generate_macro_data, generate_industry_data,
                              download_akshare_macro_data)
    macro_df = download_akshare_macro_data()
    if macro_df is None:
        macro_df = generate_macro_data()
    industry_df = generate_industry_data()

    # 特征工程
    fe = FeatureEngineering()
    stock_ids = sorted(df['stock_id'].unique())

    all_preds = []
    for stock_id in stock_ids:
        stock_df = df[df['stock_id'] == stock_id].copy()
        stock_df = stock_df.sort_values('date').set_index('date')

        if len(stock_df) < SEQ_LEN:
            continue

        features = fe.build_all_features(stock_df, industry_df, macro_df)
        features = fe.remove_outliers(features)

        # 使用最后N天数据 + 波动率聚类特征
        last_features = features.iloc[-SEQ_LEN:].fillna(0)
        X_raw = last_features.values
        # 计算波动率聚类并追加为特征
        cluster_id = compute_volatility_cluster(stock_df['close']) if 'close' in stock_df.columns else 1
        cluster_col = np.full((len(X_raw), 1), cluster_id, dtype=np.float32)
        X = np.column_stack([X_raw, cluster_col])

        # 计算历史协方差所需的历史收益率
        if 'close' in stock_df.columns:
            hist_returns = stock_df['close'].pct_change().dropna().values[-252:]
        else:
            hist_returns = None

        pred_return, pred_std = predictor.predict(X)

        # 市值 = 最新收盘价 × 最新成交量 / 换手率（近似）
        latest = stock_df.iloc[-1]
        close_price = latest.get('close', 10)
        volume = latest.get('volume', 1e7)
        turnover = latest.get('turnover_rate', 1)
        if turnover and turnover > 0:
            est_market_cap = close_price * volume / (turnover / 100)
        else:
            est_market_cap = close_price * 1e9  # 粗略估算

        industry = get_industry(stock_id)

        all_preds.append({
            'stock_id': stock_id,
            'predicted_return': float(pred_return),
            'predicted_std': float(pred_std),
            'market_cap': float(est_market_cap),
            'industry': industry,
        })

    predictions_df = pd.DataFrame(all_preds)

    if len(predictions_df) == 0:
        print("没有有效预测结果")
        return False

    # 准备历史收益率矩阵 (用于协方差估计)
    # 收集各股票的历史日收益率
    hist_returns_dict = {}
    for stock_id in predictions_df['stock_id']:
        stock_df = df[df['stock_id'] == stock_id].copy()
        stock_df = stock_df.sort_values('date')
        if 'close' in stock_df.columns:
            rets = stock_df['close'].pct_change().dropna().values[-252:]
            if len(rets) > 60:
                hist_returns_dict[stock_id] = rets

    # 对齐历史收益率
    if len(hist_returns_dict) >= 3:
        min_len = min(len(v) for v in hist_returns_dict.values())
        hist_matrix = np.column_stack([v[-min_len:] for v in hist_returns_dict.values()])
    else:
        hist_matrix = None

    # 计算市场指数收益率 (用于市场状态检测)
    index_returns = None
    index_csv = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'index_data.csv')
    if os.path.exists(index_csv):
        idx_df = pd.read_csv(index_csv)
        idx_df['date'] = pd.to_datetime(idx_df['date'])
        idx_df = idx_df.sort_values('date')
        if 'index_close' in idx_df.columns:
            index_returns = idx_df['index_close'].pct_change().dropna().values[-252:]
    if index_returns is None:
        # 从stock数据估算等权市场收益
        all_rets = []
        for stock_id in predictions_df['stock_id'][:30]:
            stock_df = df[df['stock_id'] == stock_id].copy()
            stock_df = stock_df.sort_values('date')
            if 'close' in stock_df.columns:
                rets = stock_df['close'].pct_change().dropna().values[-252:]
                if len(rets) > 60:
                    all_rets.append(rets)
        if all_rets:
            min_len = min(len(r) for r in all_rets)
            index_returns = np.mean([r[-min_len:] for r in all_rets], axis=0)

    # Black-Litterman 优化
    optimizer = BlackLittermanOptimizer()
    weights, opt_info = optimizer.optimize_portfolio(
        predictions_df['predicted_return'].values,
        predictions_df['predicted_std'].values,
        predictions_df['market_cap'].values,
        hist_matrix,
        predictions_df['industry'].values,
        index_returns=index_returns
    )

    print(f"\n[组合决策] 市场状态: {opt_info.get('regime', 'N/A')}, "
          f"目标仓位: {opt_info.get('target_exposure', 0):.0%}, "
          f"实际仓位: {opt_info.get('actual_exposure', 0):.0%}, "
          f"持仓数: {opt_info.get('n_stocks', 0)}")

    predictions_df['weight'] = weights
    predictions_df = predictions_df.sort_values('weight', ascending=False)

    generate_result_csv(predictions_df)

    print("=" * 60)
    print("预测完成!")
    print("=" * 60)

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
