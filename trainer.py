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
# ckptpath = './models/ckpts/'

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
trainSPdataloader = DataLoader(trainSPdataset, batch_size=16, shuffle=True)
testSPdataloader = DataLoader(testSPdataset, batch_size=16, shuffle=True)

# %%
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

spmodel = SequenceConvTransformer().to(device)
print(spmodel)

# %%
loss_func = nn.MSELoss()
optimizer = torch.optim.Adam(spmodel.parameters(), lr=1e-3)

lrlambda = lambda epoch: 0.5 ** epoch
lrscheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lrlambda)

# %%
def train(dataloader, model, loss_func, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (refer, query, target) in enumerate(dataloader):
        refer, query, target = refer.to(device), query.to(device), target.to(device)

        pred = model(refer, query)
        loss = loss_func(pred, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(refer)
            print(f"loss: {loss:>7f}    [{current:5d}/{size:>5d}]", end='\r')
        if batch % 100 == 0:
            print()

# %%
def test(dataloader, model, loss_func):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for refer, query, target in dataloader:
            refer, query, target = refer.to(device), query.to(device), target.to(device)
            pred = model(refer, query)
            test_loss += loss_func(pred, target).item()
    test_loss /= num_batches
    print(f"Test Error: \n Avg loss: {test_loss:>8f}\n")

    return test_loss

# %%
epochs = 300
lossarray = np.zeros((epochs), dtype=np.float32)
ckptpath = './models/ckpts/'
bestckpt = ckptpath + 'bestmodel'
bestloss = 100
for t in range(epochs):
    torch.cuda.empty_cache()
    print(f"Epoch {t+1}\n------------------------")
    print(f"Learning rate: {optimizer.param_groups[0]['lr']:>8f}")
    train(trainSPdataloader, spmodel, loss_func, optimizer)
    print()
    testloss = test(testSPdataloader, spmodel, loss_func)
    lossarray[t] = testloss
    lrscheduler.step()
    if testloss <= bestloss:
        torch.save({
            'epoch': t,
            'model_state_dict': spmodel.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': testloss,
        }, bestckpt+f'_epoch_{t:>d}_loss_{testloss:>4f}.pt')
    if t % 30 == 0:
        torch.save({
            'epoch': t,
            'model_state_dict': spmodel.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': testloss,
        }, ckptpath+f'ckpt_epoch_{t:>d}_loss_{testloss:>4f}.pt')
print("Done")
torch.save(spmodel, './models/spmodel.pth')
np.save(ckptpath+'testloss.npy', lossarray)

