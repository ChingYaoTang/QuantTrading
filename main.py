import os
#from dotenv import load_dotenv
import finlab
from finlab import data
from finlab.backtest import sim
import numpy as np
import pandas as pd

from functools import reduce
import random
import multiprocessing
from deap import base, creator, tools

#_ = load_dotenv('/content/drive/MyDrive/Colab Notebooks/Fin/API_key/.env')
#finlab.login(os.getenv('Finlab_API_KEY'))
#這裡要替換成自己的FinLab VIP token
token = 'hjSym0xjhsPkvV0pEzrELVOG+FdCbAwHb0wNhZbVzLtMBiTo16j7pcEXx9Ye784r#free'
finlab.login(token)

##############
### Data Prep.
##############
# import data from finlab
#monthly_revenue_grow = data.get("monthly_revenue:去年同月增減(%)").fillna(0)
#operating_profit = data.get("fundamental_features:營業利益率")
#net_income_grow = data.get('fundamental_features:稅後淨利成長率')
#eps = data.get('financial_statement:每股盈餘')
#stock = data.get('financial_statement:存貨')
#stock_turnover = data.get('fundamental_features:存貨週轉率').replace([np.nan, np.inf], 0)
#season_revenue = data.get('financial_statement:營業收入淨額')
#fcf_in = data.get('financial_statement:投資活動之淨現金流入_流出')
#fcf_out = data.get('financial_statement:營業活動之淨現金流入_流出')

close = data.get("price:收盤價")
close_pct_change = close.pct_change()
adj_close = data.get('etl:adj_close')
vol_stock = data.get('price:成交股數')
vol = data.get("price:成交金額")
rev = data.get('monthly_revenue:當月營收')
rev_year_growth = data.get('monthly_revenue:去年同月增減(%)')
rev_month_growth =  data.get('monthly_revenue:上月比較增減(%)')
ma5 = close.average(5)
ma20 = close.average(20)
ma60 = close.average(60)
vol_ma20 = vol.average(20)
vol_stock_ma20 = vol_stock.average(20)
RS = close_pct_change.clip(lower=0).rolling(20).sum() / abs(close_pct_change.clip(upper=0).rolling(20).sum())
RSI = 100 - 100 / (1 + RS)

############
# Processing
############
# Revenue growth
#monthly_revenue_grow_aa = ((monthly_revenue_grow > 0).sustain(6)) & (monthly_revenue_grow.average(6) > 25) & (monthly_revenue_grow.rise())
# Profit rate
#operating_profit_stable = (((operating_profit.diff() / operating_profit.shift().abs()).rolling(3).min()) * 100) > -20
#operating_profit_aa_1 = operating_profit_stable & (operating_profit.average(4) > 15)
#operating_profit_aa_2 = operating_profit_stable & ((operating_profit.average(4) <= 15) & ((operating_profit.average(4) > 10)) & operating_profit.rise())
#operating_profit_aa = operating_profit_aa_1 | operating_profit_aa_2
# Profit growth
#net_income_grow_aa_1 = ((net_income_grow > 0).sustain(3)) & (net_income_grow.rise())
#net_income_grow_aa_2 = (net_income_grow > 50).sustain(3)
#net_income_grow_aa = (net_income_grow_aa_1 | net_income_grow_aa_2)
# Profit strength
#eps_aa = (eps.rolling(4).sum() > 5) & (eps > 0)
# Inventory turnover
#stock_low = (stock_turnover <= 0) | (stock_turnover > 10) | ((stock / season_revenue) <= 0.04)
#stock_turnover_stable = (stock_turnover.diff() / stock_turnover.shift().abs()).rolling(3).min() * 100 > -20
#stock_turnover_cumulate_loss_gt_m20 = (stock_turnover.fall().sustain(3, 2)) & (stock_turnover.pct_change()[stock_turnover.pct_change() < 0].rolling(2).sum() * 100 < -20)
#stock_turnover_aa = (~stock_low) & stock_turnover_stable & (stock_turnover.average(4) > 1.5) & ~(stock_turnover_cumulate_loss_gt_m20)
# Cash flow
#fcf = (fcf_in + fcf_out)
#fcf_aa = fcf.rolling(6).min() > 0

cd1 = (((close - ma60)/ma60) > 0.05)
cd2 = (((vol_stock - vol_stock_ma20)/vol_stock_ma20) > 0.5)
cd3 = (((vol - vol_ma20)/vol_ma20) > 0.3)
cd4 = ((RSI < 70) & (RSI > 30))
cd5 = (rev_year_growth > 10)

#df_list = [monthly_revenue_grow_aa, operating_profit_aa, net_income_grow_aa, eps_aa, stock_turnover_aa, fcf_aa]
df_list = [cd1,cd2,cd3,cd4,cd5]

############
# Evaluating function
############
def eval(index_list, df_list, eval_dict):
    # Cache:
    id = ''.join([str(value) for value in index_list])
    if id in eval_dict:
        return eval_dict[id]
    else:
        # Filter the MyClass instances which are selected (index == 1)
        selected_dfs = [df for index, df in zip(index_list, df_list) if index == 1]

        # If no DataFrames are selected, return an empty DataFrame
        if not selected_dfs:
            result = pd.DataFrame()
        else:
            # Perform the intersection operation on the selected DataFrames using reduce
            result = reduce(lambda x, y: x & y, selected_dfs)

        if result.empty:
            value = 0
        else:
            metrics = sim(result, resample='W', market='TW_STOCK', upload=False).get_metrics()
            value = (metrics['profitability']['annualReturn'] / abs(metrics['risk']['maxDrawdown'])) * metrics['ratio']['sortinoRatio']

        eval_dict[id] = (value,)
        return (value,)  # Return a tuple


#######
# Main
#######
def main(Population_size, Generation_num, Threshold, df_list):
    # Initialization
    Cond_num = len(df_list)

    # Creator
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    # Toolbox
    toolbox = base.Toolbox()
    # Attribute generator
    toolbox.register("attr_bool", random.randint, 0, 1)
    # Structure initializers
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, Cond_num)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Operators
    num_cpus = multiprocessing.cpu_count() - 1
    pool = multiprocessing.Pool(processes=num_cpus)
    toolbox.register("map", pool.imap)


    # Use Manager to create a shared dictionary
    manager = multiprocessing.Manager()
    eval_dict = manager.dict()

    # Pass eval_dict to the evaluate function
    toolbox.register("evaluate", eval, df_list=df_list, eval_dict=eval_dict)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=Population_size)

    # Evaluate the initial population
    fitnesses = list(toolbox.map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    CXPB, MUTPB = 0.5, 0.2
    g = 0

    # Initialize fits list
    fits = [ind.fitness.values[0] for ind in pop]

    while max(fits) < Threshold and g < Generation_num:
        g += 1
        print("-- Generation %i --" % g)

        offspring = toolbox.select(pop, len(pop))
        offspring = list(pool.imap(toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = list(toolbox.map(toolbox.evaluate, invalid_ind))

        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        pop[:] = offspring

        fits = [ind.fitness.values[0] for ind in pop]

        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.5

        print("  Min %s" % min(fits))
        print("  Max %s" % max(fits))
        print("  Avg %s" % mean)
        print("  Std %s" % std)

    return dict(eval_dict)

#########
# Example usage:
#########
Population_size = 30
Generation_num = 5
Threshold = 30

result = main(Population_size, Generation_num, Threshold, df_list)