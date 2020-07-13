import numpy as np
import random


def read_int(fn):
  return [int(x.strip()) for x in open(fn).readlines()]


class PaddedDataIterator():
  def __init__(self, ts, fs, T, diff=False):
    self.ts = ts
    self.fs = fs
    self.T = T
    self.diff = diff
    self.size = len(self.ts)
    assert len(self.ts) == len(self.fs)
    self.length = [len(item) for item in self.ts]
    for i, l in enumerate(self.length):
      assert len(self.fs[i]) >= l
    self.epochs = 0
    self.shuffle()

  def shuffle(self):
    indices = list(range(self.size))
    random.shuffle(indices)
    self.ts = [self.ts[i] for i in indices]
    self.fs = [self.fs[i] for i in indices]
    self.length = [len(item) for item in self.ts]
    self.cursor = 0

  def next_batch(self, n):
    if self.cursor+n > self.size:
      self.epochs += 1
      self.shuffle()
    res_t = self.ts[self.cursor:self.cursor+n]
    res_f = self.fs[self.cursor:self.cursor+n]
    seqlen = self.length[self.cursor:self.cursor+n]
    self.cursor += n

    # Pad sequences with 0s so they are all the same length
    maxlen = max(seqlen)
    x = np.ones([n, maxlen, 1], dtype=np.float32)*self.T
    for i, x_i in enumerate(x):
      x_i[:seqlen[i], 0] = res_t[i]

    y = np.zeros([n, maxlen, 1], dtype=np.float32)
    for i, y_i in enumerate(y):
      y_i[:seqlen[i], 0] = res_f[i][:seqlen[i]]

    if self.diff == True:
      # x = np.concatenate([x[:, 0:1,:], np.diff(x, axis=1)], axis=1)
      x = np.concatenate(
          [np.zeros((len(x), 1, 1)), np.diff(x, axis=1)], axis=1)
    return x, y, np.asarray(seqlen)

# ts = [[1, 2, 4, 7, 11], [1, 3, 6, 10], [1, 4, 8, 13, 14, 16], [1, 5, 10, 16]]
# fs = [[1, 2, 4, 7, 11], [1, 3, 6, 10], [1, 4, 8, 13, 14, 16], [1, 5, 10, 16]]
# it = PaddedDataIterator(ts, fs, 20, True)

# for _ in range(4):
#   t, f, sl = it.next_batch(2)
#   print(t)
#   print(f)
#   print(sl)
#   print()

# exit()


class IntensityHomogenuosPoisson():

  def __init__(self, lam):
    self.lam = lam

  def getValue(self, t):
    return self.lam

  def getUpperBound(self, from_t, to_t):
    return self.lam


def generate_sample(intensity, T, n):
  Sequnces = []
  i = 0
  while True:
    seq = []
    t = 0
    while True:
      intens1 = intensity.getUpperBound(t, T)
      dt = np.random.exponential(1/intens1)
      new_t = t + dt
      if new_t > T:
        break

      intens2 = intensity.getValue(new_t)
      u = np.random.uniform()
      if intens2/intens1 >= u:
        seq.append(new_t)
      t = new_t
    if len(seq) > 1:
      Sequnces.append(seq)
      i += 1
    if i == n:
      break
  return Sequnces
