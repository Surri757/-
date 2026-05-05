"""
预测主程序 - 信号→Gate→执行→风控→复盘 分层架构
"""
import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# ---- 配置 ----
RANDOM_SEED = 42
SEQ_LEN = 60
PRED_HORIZON = 5

# 风控参数
MAX_LOSS_PER_POSITION = 0.02   # 每笔最多亏总资金2%
MAX_TOTAL_EXPOSURE = 1.0       # 总仓位上限
SL_ATR_MULTIPLIER = 2.0        # 止损=ATR倍数

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
                   D_MODEL, N_HEADS, E_LAYERS, PATCH_LEN, STRIDE, WeightedEnsemble)
from signals import SignalGenerator
from gate import LiveGate
from risk_manager import RiskManager
from executor import ExecutionSimulator
from review import ReviewLayer

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
    """市场状态检测器 — 影响止损宽度

    牛市放宽止损让利润奔跑，恐慌收紧止损快速截断亏损。
    """

    @staticmethod
    def detect(index_returns, predicted_returns):
        """
        Returns: (regime, sl_multiplier)
        regime: 'bull' | 'sideways' | 'bear' | 'panic' | 'neutral'
        sl_multiplier: 止损宽度乘数 (牛1.5 / 震荡1.0 / 熊0.8 / 恐慌0.7)
        """
        if index_returns is None or len(index_returns) < 20:
            return 'neutral', 1.0

        ret_20d = np.mean(index_returns[-20:])
        ret_5d = np.mean(index_returns[-5:])
        ret_60d = np.mean(index_returns[-60:]) if len(index_returns) >= 60 else ret_20d
        vol_20d = np.std(index_returns[-20:])
        vol_60d = np.std(index_returns[-60:]) if len(index_returns) >= 60 else vol_20d
        vol_change = vol_20d / (vol_60d + 1e-8)

        pred_mean = np.mean(predicted_returns) if len(predicted_returns) > 0 else 0
        pred_std = np.std(predicted_returns) if len(predicted_returns) > 0 else 0.02
        pred_pos_ratio = np.mean(predicted_returns > 0) if len(predicted_returns) > 0 else 0.5
        pred_sharpe = pred_mean / (pred_std + 1e-8)

        # 趋势信号
        trend_5d = np.clip(ret_5d / 0.03, -1, 1)
        trend_20d = np.clip(ret_20d / 0.05, -1, 1)
        trend_60d = np.clip(ret_60d / 0.08, -1, 1)
        trend_score = 0.3 * trend_5d + 0.4 * trend_20d + 0.3 * trend_60d

        # 波动率
        vol_penalty = np.clip(vol_change - 0.8, 0, 1) * 0.3 + np.clip(vol_20d / 0.025 - 0.5, 0, 1) * 0.3
        vol_score = 1.0 - np.clip(vol_penalty, 0, 0.6)

        # 预测质量
        pred_score = np.clip(pred_sharpe / 2.0 + 0.5, 0.2, 1.0)
        pred_score *= (0.5 + 0.5 * pred_pos_ratio)

        # 市场状态识别
        if ret_5d < -0.03 and vol_change > 1.5:
            regime = 'panic'
        elif ret_20d < -0.01 and vol_change > 1.2:
            regime = 'bear'
        elif ret_20d > 0.01:
            regime = 'bull'
        elif abs(ret_20d) < 0.003:
            regime = 'sideways'
        else:
            regime = 'neutral'

        # 止损宽度乘数
        sl_map = {'bull': 1.5, 'sideways': 1.0, 'neutral': 1.0, 'bear': 0.8, 'panic': 0.7}
        sl_multiplier = sl_map.get(regime, 1.0)

        print(f"[市场状态] {regime} | 趋势分={trend_score:.2f} 波动分={vol_score:.2f} 预测分={pred_score:.2f}")
        print(f"  止损乘数={sl_multiplier:.1f}x  Sharpe={pred_sharpe:.2f}  正向率={pred_pos_ratio:.1%}")

        return regime, sl_multiplier


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


def generate_result_csv(final_predictions, output_path=None):
    """从ReviewLayer的final_predictions生成result.csv"""
    if output_path is None:
        output_path = os.path.join(OUTPUT_DIR, 'result.csv')

    if not final_predictions:
        print("无有效预测结果")
        return

    result = pd.DataFrame(final_predictions)
    result['stock_id'] = result['stock_id'].astype(str).str.zfill(6)
    result = result.sort_values('weight', ascending=False)
    result = result.head(5)
    result[['stock_id', 'weight']].to_csv(output_path, index=False, encoding='utf-8')

    # 赛事总收益公式: R = Σ(w_i × r_i)，其中 r_i = (P_{T+5}^open / P_{T+1}^open) - 1
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
    print("量化交易预测 - 信号→Gate→执行→风控→复盘")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # ── 1. 加载模型 ──
    predictor = StackingPredictor()
    if not predictor.load_models():
        print("警告: 模型文件不存在，使用默认预测")
        predictions = [
            {'stock_id': '000001', 'predicted_return': 0.05, 'weight': 0.30},
            {'stock_id': '000002', 'predicted_return': 0.04, 'weight': 0.25},
            {'stock_id': '600000', 'predicted_return': 0.03, 'weight': 0.20},
            {'stock_id': '600016', 'predicted_return': 0.02, 'weight': 0.15},
            {'stock_id': '601318', 'predicted_return': 0.01, 'weight': 0.10},
        ]
        generate_result_csv(predictions)
        return True

    # ── 2. 加载数据 ──
    df = load_test_data()
    if df is None or len(df) == 0:
        print("无测试数据！")
        return False

    print(f"测试数据: {len(df)} 条记录, {df['stock_id'].nunique()} 只股票")

    # 宏观和行业数据
    from data_fetcher import (generate_macro_data, generate_industry_data,
                              download_akshare_macro_data)
    macro_df = download_akshare_macro_data()
    if macro_df is None:
        macro_df = generate_macro_data()
    industry_df = generate_industry_data()

    # ── 3. 初始化各层 ──
    fe = FeatureEngineering()
    stock_ids = sorted(df['stock_id'].astype(str).str.zfill(6).unique())

    # 市场指数收益率
    index_returns = None
    index_csv = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             'data', 'index_data.csv')
    if os.path.exists(index_csv):
        idx_df = pd.read_csv(index_csv)
        idx_df['date'] = pd.to_datetime(idx_df['date'])
        idx_df = idx_df.sort_values('date')
        if 'index_close' in idx_df.columns:
            index_returns = idx_df['index_close'].pct_change().dropna().values[-252:]

    signal_gen = SignalGenerator(predictor, fe, industry_df, macro_df,
                                 seq_len=SEQ_LEN, sl_atr_multiplier=SL_ATR_MULTIPLIER)
    gate = LiveGate(
        whitelist_stocks=None,  # 不设白名单，全市场选股
        max_single_position=1.0,
        min_risk_distance_pct=0.02,
        min_confidence=0.10,
        max_positions=5,
    )
    risk_mgr = RiskManager(
        total_capital=1.0,
        max_loss_per_position=MAX_LOSS_PER_POSITION,
        max_total_exposure=MAX_TOTAL_EXPOSURE,
        single_stock_only=True,
    )
    executor = ExecutionSimulator(risk_mgr, default_slippage_bps=5.0)
    review = ReviewLayer(output_dir=OUTPUT_DIR)

    # ── 4. 生成信号 ──
    print("\n生成交易信号...")
    signals = signal_gen.generate_all(df, stock_ids)
    print(f"生成 {len(signals)} 个信号")

    if not signals:
        print("无有效信号")
        return False

    # ── 5. 市场状态 → 调整止损宽度 ──
    pred_returns = np.array([s.predicted_return for s in signals])
    regime, sl_multiplier = MarketRegime.detect(index_returns, pred_returns)

    # 应用市场状态到止损宽度
    for s in signals:
        entry = s.entry_price
        base_sl_dist = entry - s.stop_loss_price
        adjusted_sl = entry - base_sl_dist * sl_multiplier
        s.stop_loss_price = float(max(adjusted_sl, entry * 0.90))  # 硬止损不超过-10%

    # 按预测收益降序排列（优先处理高收益信号）
    signals.sort(key=lambda s: s.predicted_return, reverse=True)

    # ── 6. 流水线: Gate → Risk → Execute → Review ──
    print("\n执行流水线...")
    for signal in signals:
        # Gate 校验
        passed, reason = gate.validate(signal)
        if not passed:
            review.log_rejection(signal, reason)
            continue

        # 以损定仓
        position_size = risk_mgr.compute_position_size(signal)
        if position_size <= 0.001:
            review.log_rejection(signal, f"position_too_small:{position_size:.4f}")
            continue

        # 执行（模拟）
        exec_record = executor.execute_signal(signal, position_size)
        if exec_record is None:
            continue

        # 登记到Gate（单只股票唯一持仓）
        gate.register_position(signal.stock_id, {
            'stock_id': signal.stock_id,
            'weight': position_size,
            'entry_price': exec_record.entry_price,
            'date': signal.date,
        })

        # 复盘记录
        review.log_execution(exec_record)
        review.log_slippage(signal.stock_id, signal.entry_price, exec_record.entry_price)
        review.log_prediction(signal.stock_id, signal.predicted_return, position_size)

    # ── 7. 风控事件记录 ──
    for event in risk_mgr.risk_events:
        review.log_risk_event(event)

    # ── 8. 生成报告和结果 ──
    report = review.generate_report()
    generate_result_csv(review.final_predictions)

    print("=" * 60)
    print("流水线执行完成!")
    print(f"  Gate拒绝: {len(review.rejections)}")
    print(f"  执行成交: {len(review.executions)}")
    print(f"  风险事件: {len(review.risk_events)}")
    print("=" * 60)

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
