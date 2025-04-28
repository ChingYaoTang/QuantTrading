# Evaluating function

import pandas as pd
from functools import reduce

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