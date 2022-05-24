# %%
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from SPModel import ReferEncoder
from SPModel import QueryEncoder
from SPModel import SequenceConvTransformer
from dataload import SPDataset

# %%
trainnpdir = '../input/train_files/processed/'
train_sp_dir = trainnpdir+'train_sp_nonan.npy'

ReferLen = 1200
QueryLen = 60
QueryDelay = 20
Datalength = ReferLen + QueryDelay

train_ratio = 0.8

SPdataset = SPDataset(train_sp_dir)
datasettotallen = SPdataset.__len__()
trainlen = int(train_ratio*datasettotallen)
testlen = datasettotallen - trainlen
trainSPdataset, testSPdataset = torch.utils.data.random_split(SPdataset, [trainlen, testlen])

# %%
testSPdataloader = DataLoader(testSPdataset, batch_size=1, shuffle=True)

# %%
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

spmodel = SequenceConvTransformer().to(device)

# %%
ckptpath = './models/ckpts/bestmodel_epoch_26_loss_0.000507.pt'
ckpt = torch.load(ckptpath, map_location=torch.device(device))
spmodel.load_state_dict(ckpt['model_state_dict'])

# %%
testnum = 0
correctcount = 0
meanerrorrate = 0
targetnozeronum = 0
for i, (refer, query, target) in enumerate(testSPdataloader):
    testnum += 1
    refer, query, target = refer.to(device), query.to(device), target.to(device)
    spmodel.eval()
    pred = spmodel(refer, query).item()
    target = target.item()
    if pred * target >= 0:
        correctcount += 1
    if target != 0:
        targetnozeronum += 1
        meanerrorrate += abs(target-pred) / abs(target)

    print(i, 'Prediction:', pred, "Target:", target, end='\r')

    # if i >= testnum-1:
    #     meanerrorrate /= testnum
    #     print()
    #     break
meanerrorrate /= targetnozeronum
print()
print("Correct num:", correctcount, "Total num:", testnum, "Precision:", correctcount/testnum)
print("Mean Error rate:", meanerrorrate)


