import os
import sys
from math import ceil, floor, pi

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from tqdm import trange, tqdm

import utils

BATCH_SIZE = 16
HIDDEN_SIZE = 32
TRAIN_FRAC = 0.8
N_EPOCHS = 2


class DMPP(nn.Module):
  def __init__(self, t_rep, s_rep, batch_size, max_train_t, hidden_size=32):
    super().__init__()

    self.eps = 1e-8
    T_len = len(t_rep)
    S_len = len(s_rep)
    # self.T_len = len(t_rep)
    # self.S_len = len(s_rep)
    # self.J = self.T_len*self.S_len

    t_ = np.transpose(np.tile(t_rep, (S_len, 1)))
    s_ = np.tile(s_rep, (T_len, 1))
    rep = np.concatenate([np.expand_dims(t_, axis=-1),
                          np.expand_dims(s_, axis=-1)], axis=-1)
    print('model.rep:', rep.shape)
    rep = rep.reshape(-1, 2)
    self.rep = torch.tensor(rep, dtype=torch.float32, requires_grad=False)
    self.rep = self.rep.cuda()
    print('model.rep:', self.rep.shape)
    # self.rep_mod = self.rep.view(1, self.rep.shape[0], 2)
    # self.rep_mod = self.rep_mod.repeat(batch_size, 1, 1)
    # print('model.rep_mod:', self.rep_mod.shape)

    self.rep_test = self.rep[self.rep[:, 0] > max_train_t]
    print('model.rep_test:', self.rep_test.shape)
    # self.rep_test_mod = self.rep_test.view(1, self.rep_test.shape[0], 2)
    # self.rep_test_mod = self.rep_test_mod.repeat(batch_size, 1, 1)
    # print('model.rep_test_mod:', self.rep_test_mod.shape)
    print()

    # sigma = torch.rand(2, 2, dtype=torch.float32, requires_grad=True)
    # sigma = torch.mm(sigma, sigma.t())
    # self.sigma = nn.Parameter(sigma+torch.eye(2))
    self.sigma_ = torch.rand(2, 2, dtype=torch.float32, requires_grad=True)
    self.sigma_ = self.sigma_.cuda()

    self.fcc = nn.Linear(1, hidden_size)
    self.final_fcc = nn.Linear(hidden_size, 1)
    self.f = nn.Sequential(
        self.fcc,
        nn.ReLU(),
        self.final_fcc,
        nn.ReLU()
    )

    # nn.init.xavier_normal_(self.fcc.weight)
    # nn.init.xavier_normal_(self.final_fcc.weight)

  def sigma(self):
    # return a positive semi definite version of sigma
    return torch.mm(self.sigma_, self.sigma_.t())

  def kernel(self, x, u):
    a = x-u
    an = torch.norm(a, dim=2)
    a_ = a[an > 1000]
    power = torch.matmul(
        torch.matmul(a_.unsqueeze(-2), self.sigma()),
        a_.unsqueeze(-1))
    b = torch.zeros_like(an)
    b[an > 1000] = power.squeeze()
    return torch.exp(-b)

  def log_likelihood(self, mini_batch, N, is_test=False):
    # Following eqn 7 in paper.

    # mini_batch.shape (bs x 2)
    bs, fsize = mini_batch.shape

    if is_test:
      rep = self.rep_test
      # rep_mod = self.rep_test_mod
    else:
      rep = self.rep
      # rep_mod = self.rep_mod
    J = rep.shape[0]

    rep_mod = rep.view(1, rep.shape[0], 2)
    rep_mod = rep_mod.repeat(bs, 1, 1)

    # convert both input and representative points to (bs x J x 2)
    mb = mini_batch.view(bs, 1, fsize).repeat(1, J, 1)
    # print('J:', J)
    # print('mb.shape:', mb.shape)
    # print('rep_mod.shape:', rep_mod.shape)
    # print()

    # kernel value, for i in mini batch, j in J
    k = self.kernel(mb, rep_mod)  # bs x J

    # feature function
    f = self.f(rep[:, 1:2])  # J x 1
    # multiply all cols of k by f, element wise
    fk = f.squeeze()*k  # (J) * (bs x J) => (bs x J)

    print('fk sum:', torch.sum(fk, dim=1))
    print('fk log:', torch.log(torch.sum(fk, dim=1)))
    print('det:', torch.det(self.sigma()))
    # # print('fk log + eps:', torch.log(torch.sum(fk, dim=1) + self.eps))

    # first part of eqn 7
    res = (N*torch.sum(torch.log(torch.sum(fk, dim=1) + self.eps)))/bs
    print('res1:', res)
    # second part of eqn 7
    res -= pi*torch.sum(f) / torch.sqrt(torch.det(self.sigma()))
    print('res :', res)
    return res


# def main():
folder = 'datasets/web_server/epa'
ts = utils.read_int(os.path.join(folder, 'timestamp.txt'))
fs = utils.read_int(os.path.join(folder, 'frequency.txt'))

ts_min = min(ts)
ts_max = max(ts)
ts = [x-ts_min for x in ts]

ts_train_max = ts[int(len(ts)*TRAIN_FRAC)]

L = max(fs)
# if L > 100:
#   L = 100

data = np.array(list(zip(ts, fs)))
train_data = data[data[:, 0] <= ts_train_max]
test_data = data[data[:, 0] > ts_train_max]

N_train = len(train_data)
N_test = len(test_data)

print('N_train:', N_train)
print('N_test:', N_test)
print()

Tmin = 0
Tmax = 24*60*60
# avg inter arrival time as time diff between each representative point
t_diff = ceil(np.mean(np.diff(ts)))
N_Trep = floor(Tmax/t_diff)+1

Trep = list(range(0, t_diff*N_Trep, t_diff))
Frep = list(range(1, L+1))

print('len(Trep):', len(Trep))
print('len(Frep):', len(Frep))
print()

del ts
del fs
del data

train_data = torch.tensor(train_data, dtype=torch.float32).cuda()
test_data = torch.tensor(test_data, dtype=torch.float32).cuda()

train_dataset = TensorDataset(train_data)
train_sampler = RandomSampler(train_dataset)
train_dataloader = DataLoader(
    train_dataset, sampler=train_sampler, batch_size=BATCH_SIZE,)
# train_iter = iter(train_dataloader)

test_dataset = TensorDataset(test_data)
test_sampler = SequentialSampler(test_dataset)
test_dataloader = DataLoader(
    test_dataset, sampler=test_sampler, batch_size=BATCH_SIZE)

model = DMPP(Trep, Frep, max_train_t=ts_train_max,
             batch_size=BATCH_SIZE, hidden_size=HIDDEN_SIZE)
model = model.cuda()
optimizer = Adam(model.parameters(), lr=1e-4)

print('Starting training')

num_train_iters = ceil(N_train/BATCH_SIZE)
num_test_iters = ceil(N_test/BATCH_SIZE)
log_interval = ceil(N_train/BATCH_SIZE)//5  # n times per batch
model.train()

print(model.sigma())

with tqdm(total=N_EPOCHS*num_train_iters) as pbar:
  for epoch in range(N_EPOCHS):
    print('EPOCH:', epoch)

    for it, data in enumerate(train_dataloader):
      # data = next(train_iter)[0]
      loss = -model.log_likelihood(data[0], N_train)
      # print(loss.item())

      model.zero_grad()
      loss.backward()
      clip_grad_norm_(parameters=model.parameters(), max_norm=5.0)
      optimizer.step()

      if it == 0 or (it+1) % log_interval == 0:
        model.eval()
        with torch.no_grad():
          log_likes = []
          for test_batch in test_dataloader:
            log_like = model.log_likelihood(
                test_batch[0], N_test, is_test=True)
            log_likes.append(log_like.item())
            # print(log_like.item())
          print('it:{} test data likelihood: {}'.format(it, np.mean(log_likes)))
          print(model.sigma())
        model.train()

      pbar.update(1)


# if __name__ == '__main__':
#   main()
