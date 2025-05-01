# download_and_cache_data.py (更新版)

import os
import pickle
from finlab import login, data

token = 'grQpDN7cYHEPelP/MfrHwj3s3S/pl6lpoxJ8zE985wGWrmTCmhAhbjPDnlhR2IfO#free'
login(token)

# 所有資料鍵與對應 FinLab key
DATA_KEYS = {
    'market_value': 'etl:market_value',
    'close': 'price:收盤價',
    'adj_close': 'etl:adj_close',
    'vol_stock': 'price:成交股數',
    'vol': 'price:成交金額',
    'rev': 'monthly_revenue:當月營收',
    'rev_year_growth': 'monthly_revenue:去年同月增減(%)',
    'rev_month_growth': 'monthly_revenue:上月比較增減(%)',

    # 回測必備行情資料
    'open': 'price:開盤價',
    'high': 'price:最高價',
    'low': 'price:最低價',
    'close_price': 'price:收盤價',  # 跟 close 一樣，保險起見再存一次
}

os.makedirs('local_finlab_data', exist_ok=True)

for name, key in DATA_KEYS.items():
    print(f'Fetching {name} from {key}...')
    df = data.get(key)
    with open(f'local_finlab_data/{name}.pkl', 'wb') as f:
        pickle.dump(df, f)
    print(f'Saved to local_finlab_data/{name}.pkl')

print('✅ 所有資料已下載完畢並存至本地端。')
