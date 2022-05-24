# %%
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np

# %%
trainnpdir = '../input/train_files/processed/'
train_sp_dir = trainnpdir+'train_sp_nonan.npy'

# %%
ReferLen = 1200
QueryLen = 60
QueryDelay = 20
Datalength = ReferLen + QueryDelay

train_ratio = 0.8

# %%
class SPDataset(Dataset):
    def __init__(self, train_sp_dir):
        self.train_sp_whole = np.load(train_sp_dir)
        print(np.sum(np.isnan(self.train_sp_whole)), self.train_sp_whole.shape)
        datelen = self.train_sp_whole.shape[2]
        stocknum = self.train_sp_whole.shape[0]
        sliceforonestock = datelen - Datalength
        self.dataset_np = np.zeros((stocknum*sliceforonestock, 8, Datalength), dtype=np.float64)
        for i in range(stocknum):
            for j in range(sliceforonestock):
                startdate = self.train_sp_whole[i, 0, 0]
                self.dataset_np[i*sliceforonestock+j, 0:7, :] = self.train_sp_whole[i, 0:7, j:j+Datalength]
                self.dataset_np[i*sliceforonestock+j, 7, :] = self.train_sp_whole[i, 9, j:j+Datalength]
                self.dataset_np[i*sliceforonestock+j, 0, :] -= startdate
        print(np.sum(np.isnan(self.dataset_np)), self.dataset_np.shape)
        
    def __len__(self):
        return self.dataset_np.shape[0]

    def __getitem__(self, idx):
        data_np = self.dataset_np[idx, :, :]
        refer_np = data_np[0:7, 0:1200]
        query_np = data_np[0:7, -61:-1]
        target = data_np[7, -1]

        refer = torch.from_numpy(refer_np).float()
        query = torch.from_numpy(query_np).float()
        target = torch.tensor(target).unsqueeze(0).float()
        
        return refer, query, target


# # %%
# SPdataset = SPDataset(train_sp_dir)
# datasettotallen = SPdataset.__len__()
# trainlen = int(train_ratio*datasettotallen)
# testlen = datasettotallen - trainlen
# trainSPdataset, testSPdataset = torch.utils.data.random_split(SPdataset, [trainlen, testlen])

# # %%
# trainSPdataloader = DataLoader(trainSPdataset, batch_size=64, shuffle=True)
# testSPdataloader = DataLoader(testSPdataset, batch_size=64, shuffle=True)
