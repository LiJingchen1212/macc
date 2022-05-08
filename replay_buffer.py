from numpy import random
from collections import namedtuple
import collections
import numpy as np

Transition = namedtuple('Transition',   # 默认经验元组格式
                        ('state', 'action', 'next_state', 'reward', 'done'))

# 经验回放缓存池
class ReplayBuffer(object):
    def __init__(self, capacity, kx, transition=Transition):    # 可自行经验创建元组格式
        self.capacity = capacity
        self.memory = []
        self.weight = []
        self.position = 0
        self.kx = kx
        self.transition = transition

    # 存入经验
    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
            self.weight.append(None)
        tr = self.transition(*args)
        self.memory[self.position] = tr
        self.weight[self.position] = sum(tr.reward)+self.kx
        self.position = (self.position + 1) % self.capacity
        return tr

    # 随机经验采样
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def weighted_sample(self, batch_size):
        temp = np.array(self.weight)
        p = (temp/temp.sum()).tolist()
        sam = random.choice(len(p), batch_size, p)
        tr = [self.memory[i] for i in sam]
        #return random.choice(self.memory, batch_size,  p)
        return tr

    def claer(self):
        self.memory = []
        self.position = 0

    def __len__(self):
        return len(self.memory)


class TimeAbstractMemory(object):
    def __init__(self, capacity, batchlenth, kx, gamma, transition=Transition):  # 可自行经验创建元组格式
        self.capacity = capacity
        self.memory = []
        self.weight = []
        self.gamma = gamma
        self.discount = 1.0
        self.w = 0
        self.kx = kx
        self.batchlenth = batchlenth
        self.buff = ReplayBuffer(self.batchlenth, kx, transition=transition)
        self.position = 0
        self.transition = transition

    # 存入经验
    def push(self, *args):
        tr = self.buff.push(*args)
        done = tr.done[0]
        self.w += sum(tr.reward)*self.discount
        self.discount *= self.gamma
        #将一整个episode添加进经验池
        if(done or len(self.buff)>=self.batchlenth):
            self.save_batch(self.buff)

    def save_batch(self, buff):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
            self.weight.append(None)
        self.memory[self.position] = buff
        self.buff = ReplayBuffer(self.batchlenth,self.kx, transition=self.transition)
        self.weight[self.position] = self.w/self.batchlenth + self.kx
        self.w = 0
        self.discount = 1.0
        self.position = (self.position + 1) % self.capacity

    # 随机经验采样
    def sample(self, batch):
        return random.sample(self.memory, batch)

    def weighted_sample(self, batch_size):
        temp = np.array(self.weight)
        p = (temp/temp.sum()).tolist()
        sam = random.choice(len(p), batch_size, p)
        tr = [self.memory[i] for i in sam]
        #return random.choice(self.memory, batch_size,  p)
        return tr


    def clear(self):
        self.memory = []
        self.position = 0

    def __len__(self):
        return len(self.memory)


