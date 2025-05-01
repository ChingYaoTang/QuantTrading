# 這段放在你的主程式一開始
import pickle
import numpy as np
import pandas as pd
from deap import base, creator, tools
from functools import reduce
import finlab
import random
import os
import pickle
from tqdm import tqdm
from finlab import data
from finlab.backtest import sim
from finlab.dataframe import FinlabDataFrame

def load_local_data(name):
    with open(f'local_finlab_data/{name}.pkl', 'rb') as f:
        return pickle.load(f)

# 取代原本的 data.get(...) 呼叫
市值 = load_local_data('market_value')
close = load_local_data('close')
close_pct_change = close.pct_change()
adj_close = load_local_data('adj_close')
vol_stock = load_local_data('vol_stock')
vol = load_local_data('vol')
rev = load_local_data('rev')
rev_year_growth = load_local_data('rev_year_growth')
rev_month_growth = load_local_data('rev_month_growth')

# 載入行情資料
high = load_local_data('high')
low = load_local_data('low')
open_price = load_local_data('open')
close_price = load_local_data('close_price')  # 或直接用 close


ma5 = close.average(5)
ma20 = close.average(20)
ma60 = close.average(60)
vol_ma20 = vol.average(20)
vol_stock_ma20 = vol_stock.average(20)
RS = close_pct_change.clip(lower=0).rolling(20).sum() / abs(close_pct_change.clip(upper=0).rolling(20).sum())
RSI = 100 - 100 / (1 + RS)

dataset = {
    '價格動量5': ((close - ma5)/ma5) > 0.05,
    '價格動量20': ((close - ma20)/ma20) > 0.05,
    '價格動量60': ((close - ma60)/ma60) > 0.05,
    '成交股數顯著增加': ((vol_stock - vol_stock_ma20)/vol_stock_ma20) > 0.5,
    '成交金額顯著增加': ((vol - vol_ma20)/vol_ma20) > 0.3,
    '價格波動率低': close_pct_change.rolling(20).std() < 0.02,
    '價格波動率高': close_pct_change.rolling(20).std() > 0.05,
    'RSI': (RSI < 70) & (RSI > 30),
    '月營收年增率10': rev_year_growth > 10,
    '月營收月增率20': rev_month_growth > 20,
    '交易活絡': (vol / 市值) > 0.002,
    '過去5天資金流入': (((close - close.shift(1)) * vol_stock).rolling(5).sum()) > 0,
    '價格突破20日高點': (close / close.shift(1).rolling(20).max()) > 1.02  ,
    '成交量突破20日高點': (vol_stock / vol_stock.shift(1).rolling(20).max()) > 1.5,
    '價格成交量背離' : (np.sign(close - close.shift(5)) * np.sign(vol_stock - vol_stock.shift(5))) >= 0,
    '月營收季節因子' : (rev / ((rev.shift(12) + rev.shift(24) + rev.shift(36))/3)) > 1.1,
    'MA反轉' : (ma5/ma20) > 1,
    '股價營收敏感度' : ((close - close.shift(20)) / (rev - rev.shift(1))) > 0.1,
}

# 將所有的資料轉換成boolean，並確保index是datetime，並且轉換成FinlabDataFrame
for key, value in dataset.items():
    if value.index.dtype == 'O':
        dataset[key] = value.deadline().fillna(False).astype(bool)
    else:
        dataset[key] = value.fillna(False).astype(bool)
    dataset[key] = FinlabDataFrame(dataset[key])

# 把條件數目作為基因長度
individual_size = len(dataset)
# 把dataset的key取出來，之後會用到
keys = list(dataset.keys())

#讀檔用的
def load_data(filename, default_data):
    print(f'讀取 {filename}.pkl...')
    if os.path.exists(filename + '.pkl'):
        with open(filename + '.pkl', 'rb') as f:
            print(f'{filename}.pkl 讀取成功')
            return pickle.load(f)
    print(f'{filename}.pkl 不存在，將使用預設值')
    return default_data
#存檔用的
def save_data(data, filename):
    print(f'保存 {filename}.pkl...')
    with open(filename + '.pkl', 'wb') as f:
        pickle.dump(data, f)
    print(f'{filename}.pkl 保存成功')
# 對選定的基因取交集
def cross_condition(conditions, weight):
    cond_temp = [conditions[i] for i in weight]
    return reduce(lambda i, j: i & j, cond_temp)
# 把基因轉換成條件
def gene_to_RNA(gene, keys):
    return [key for key, g in zip(keys, gene) if g == 1]
# 把條件轉換成基因
def RNA_to_gene(RNA, keys):
    return [1 if key in RNA else 0 for key in keys]

# 設定演算法要求最大值
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
# 設定個體的基因格式為List，並且包含FitnessMax的屬性
creator.create("Individual", list, fitness=creator.FitnessMax)
# 設定toolbox，工具箱用於註冊和管理演算法中使用的各種函數和操作
toolbox = base.Toolbox()
#註冊一個名為"attr_bool"的屬性生成器。使用Python的random.randint函數，生成0或1的隨機整數。
toolbox.register("attr_bool", random.randint, 0, 1)
#註冊一個用於創建個體的函數。使用tools.initRepeat函數重複調用attr_bool函數來創建一個個體。
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=individual_size)
#註冊一個用於創建整個族群的函數。使用tools.initRepeat函數重複調用toolbox.individual函數來創建一個族群。
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# 用個體基因計算適應度
def objective(gene,keys):
    RNA = gene_to_RNA(gene,keys)
    position_temp = cross_condition(dataset.copy(), RNA)

    # 檢查每年平均交易次數
    每年平均交易次數 = (abs(position_temp.diff())>0).sum().sum()/(float((position_temp.index.max() - position_temp.index.min()).days//365))
    if 每年平均交易次數 < 50:
        print(f"交易次數過少，每年平均交易次數: {每年平均交易次數}")
        return (0,)

    # report = sim(position_temp, resample='M', upload=False, position_limit=1/5, fee_ratio=1.425/1000/3,  trade_at_price='high_low_avg')
    report = sim(
        position_temp,
        resample='M',
        upload=False,
        position_limit=1/5,
        fee_ratio=1.425/1000/3,
        trade_at_price='high_low_avg',
        open_price=open_price,
        high_price=high,
        low_price=low,
        close_price=close_price
    )
    rs = report.get_stats()
    年化報酬率 = round(rs['cagr'] * 100, 3)
    最大回落 = round(rs['max_drawdown'] * 100, 3)
    日索提諾 = round(rs['daily_sortino'], 3)
    fittness = 年化報酬率/abs(最大回落) * 日索提諾
    print(f'{RNA}')
    print(f'年化報酬率: {年化報酬率}%, 最大回落: {最大回落}%, 日索提諾: {日索提諾}, 適應度: {fittness}')
    return (fittness,)

# 註冊一個用於評估整個族群的函數。使用objective函數來評估個體。
toolbox.register("evaluate", objective)
# 註冊一個用於交叉的函數。使用tools.cxUniform函數來實現均勻交叉。
toolbox.register("mate", tools.cxUniform)
# 註冊一個用於突變的函數。使用tools.mutFlipBit函數來實現基因突變。
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
# 註冊一個用於選擇的函數。使用tools.selTournament函數來實現錦標賽選擇。
toolbox.register("select", tools.selTournament, tournsize=5)

#評估整個全體的function
def evaluate(offspring, pop, evaled_ind, evaled_ind_empty, keys):
    dict_count = 0
    empty_count = 0
    for idx,ind in enumerate(offspring):
        #先將基因轉換成條件
        RNA = str(gene_to_RNA(ind,keys))
        #如果該個體的適應度還沒被計算過，則計算適應度
        if not ind.fitness.valid:
            #如果該個體的適應度已經被計算過，則直接取用
            if RNA in evaled_ind:
                ind.fitness.values = (evaled_ind[RNA],)
                offspring[idx] = ind
                dict_count += 1
                print(f'使用字典中的適應度:{evaled_ind[RNA]}')
            #如果該個體的適應度為0，則直接取用
            elif RNA in evaled_ind_empty:
                ind.fitness.values = (0,)
                offspring[idx] = ind
                empty_count += 1
                print('個體適應度為0')
            #否則計算適應度
            else:
                fit = objective(ind, keys)
                ind.fitness.values = fit
                if fit[0] > 0:
                    evaled_ind[RNA] = fit[0]
                else:
                    evaled_ind_empty.append(RNA)
    print('使用字典中的適應度數量:{}'.format(dict_count))
    print('個體適應度為0數量:{}'.format(empty_count))

    return tools.selBest(pop + offspring, len(pop)), evaled_ind, evaled_ind_empty

# 根據經驗，全亂數的個體不容易找到好的解，所以先創建有一定機率為1的個體
def create_population(n):
    print('重新創建pop...')
    population = []
    for _ in range(n):
        individual = np.zeros(individual_size, dtype=int)
        # 隨機選擇5~10個基因為1
        buy_count = np.random.randint(5, 11)
        buy_indices = random.sample(range(len(dataset)), buy_count)
        for idx in buy_indices:
            individual[idx] = 1
        individual = creator.Individual(individual.tolist())
        population.append(individual)
    return population

#設定族群大小
pop_size = 20
#打算演化多少代
gen_limit = 20
#讀取之前的存檔，沒有就創建新的族群
pop = load_data('pop', create_population(pop_size))
#讀取存檔
best_fitness = load_data('best_fitness', [])
evaled_ind = load_data('evaled_ind', {})
evaled_ind_empty = load_data('evaled_ind_empty', [])
data.use_local_data_only = True # 原本是ｆａｌｓｅ

try:
    gen = 0
    with tqdm(desc="Generations", unit="gen") as pbar:
        #判斷甚麼時候停止
        while gen<gen_limit:
            if gen >=1:
              #減少網路依賴，避免長久運行跳掉。
              data.use_local_data_only = True
            offspring = toolbox.select(pop, len(pop))
            offspring = list(map(toolbox.clone, offspring))
            #把族群按單雙分開，然後彼此有70%的機會交配
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < 0.7:
                    toolbox.mate(child1, child2, 0.5)
                    del child1.fitness.values
                    del child2.fitness.values
            #所有的後代有30%的機會進行突變
            for mutant in offspring:
                if random.random() < 0.3:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values
            #把整個後代群送進evaluate去評分
            pop, evaled_ind, evaled_ind_empty = evaluate(offspring, pop, evaled_ind, evaled_ind_empty, keys)

            #看看第一名考幾分
            max_fit = max([ind.fitness.values[0] for ind in pop])
            best_fitness.append(max_fit)

            save_data(best_fitness, 'best_fitness')
            save_data(pop, 'pop')
            save_data(evaled_ind, 'evaled_ind')
            save_data(evaled_ind_empty, 'evaled_ind_empty')

            # 更新 tqdm 的描述
            tqdm.write(f'Generation {gen} completed, Max Fitness: {max_fit:.4f}')

            # 更新進度條
            pbar.update(1)
            gen += 1
#萬一出問題了，趕快存檔。比較好debug。
except :
    save_data(pop, 'pop')
    save_data(evaled_ind, 'evaled_ind')
    save_data(evaled_ind_empty, 'evaled_ind_empty')
    save_data(best_fitness, 'best_fitness')
    raise

import matplotlib.pyplot as plt
plt.plot(best_fitness)

evaled_ind_list = list(evaled_ind.items())
evaled_ind_list.sort(key=lambda x: x[1], reverse=True)
for i in evaled_ind_list[:10]:
    print(i)