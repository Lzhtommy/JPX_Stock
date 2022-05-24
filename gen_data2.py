import numpy as np
import pandas as pd
from datetime import datetime

traindir = '../input/train_files/'
testdir = '../input/example_test_files/'
supplydir = '../input/supplemental_files/'

sp_np = np.load(traindir+'processed/sp_date_sup.npy')
print(sp_np.shape)
sp_np = np.swapaxes(sp_np, 1, 2)
print(sp_np.shape)
np.save(traindir+'processed/sp_date_sup.npy', sp_np)

# sp_sup_df = pd.read_csv(supplydir+'stock_prices.csv')
# sp_sup_df

# dates = sp_sup_df['Date'].unique()
# inde = ['Datediff', 'Open', 'High', 'Low', 'Close', 'Volume', 'AdjustmentFactor', 'ExpectedDividend', 'SupervisionFlag', 'Target']
# stocks = sp_sup_df['SecuritiesCode'].unique()
# rowidset = set(sp_sup_df['RowId'].values)
# stockcount = 0
# startdate = datetime.strptime(dates[0], r'%Y-%m-%d').date()
# npout = np.zeros((stocks.size, dates.size, 10), dtype=np.float64)
# # npout = np.zeros((1, dates.size, 10), dtype=np.float)
# for stock in stocks:
#     datecount = 0
#     for date in dates:
#         datestr = ''.join(list(date.split('-')))
#         rowid = datestr + '_' + str(stock)
#         datediff = (datetime.strptime(date, r'%Y-%m-%d').date()-startdate).days
#         npout[stockcount][datecount][0] = datediff
#         if rowid in rowidset:
#             thisrow = sp_sup_df.loc[sp_sup_df['RowId'] == rowid][inde[1:]]
#             lastrow = thisrow
#             for i in range(1,10):
#                 npout[stockcount][datecount][i] = thisrow[inde[i]]
#         else:
#             for i in range(1,10):
#                 npout[stockcount][datecount][i] = lastrow[inde[i]]
#         datecount += 1
#     stockcount += 1
#     print('Stock Count: '+ str(stockcount), end = '\r', flush=True)
#     # if stockcount > 0:
#     #     break
# np.save(traindir+'processed/sp_date_sup.npy', npout)
# print(npout.shape)