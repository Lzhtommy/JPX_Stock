# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import ipykernel
from datetime import datetime
import sys

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

traindir = '../input/train_files/'
testdir = '../input/example_test_files/'
sp_train_df = pd.read_csv(traindir+'stock_prices.csv')
sp_train_df

dates = sp_train_df['Date'].unique()
inde = ['Datediff', 'Open', 'High', 'Low', 'Close', 'Volume', 'AdjustmentFactor', 'ExpectedDividend', 'SupervisionFlag', 'Target']
stocks = sp_train_df['SecuritiesCode'].unique()
rowidset = set(sp_train_df['RowId'].values)
stockcount = 0
startdate = datetime.strptime(dates[0], r'%Y-%m-%d').date()
npout = np.zeros((stocks.size, dates.size, 10), dtype=np.float64)
# npout = np.zeros((1, dates.size, 10), dtype=np.float)
for stock in stocks:
    datecount = 0
    for date in dates:
        datestr = ''.join(list(date.split('-')))
        rowid = datestr + '_' + str(stock)
        datediff = (datetime.strptime(date, r'%Y-%m-%d').date()-startdate).days
        npout[stockcount][datecount][0] = datediff
        if rowid in rowidset:
            thisrow = sp_train_df.loc[sp_train_df['RowId'] == rowid][inde[1:]]
            lastrow = thisrow
            for i in range(1,10):
                npout[stockcount][datecount][i] = thisrow[inde[i]]
        else:
            for i in range(1,10):
                npout[stockcount][datecount][i] = lastrow[inde[i]]
        datecount += 1
    stockcount += 1
    print('Stock Count: '+ str(stockcount), end = '\r', flush=True)
    # if stockcount > 0:
    #     break
np.save(traindir+'processed/sp_date.npy', npout)
print(npout.size)