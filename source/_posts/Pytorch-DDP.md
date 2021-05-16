---
title: Pytorch DDP
date: 2021-05-5 11:06:35
tags:
    - 深度学习
---


## DDP 原理

- 在分类上，DDP 属于 Data Parallel。简单来说，就是通过提高 batch size 来增加并行度。
- 通过 Ring-Reduce 的数据交换方法提高了通信效率，并通过启动多个进程的方式减轻 Python GIL 的限制，从而提高训练速度。

<!-- more -->

假如有 N 张显卡

1. 在 DDP 模式下，会有 N 个进程被启动，每个进程在一张卡上加载一个模型，这些模型的参数在数值上是相同的。
2. （Ring-Reduce 加速）在模型训练时，各个进程通过 Ring-Reduce 方法与其他进程通信，交换各自的梯度，从而获得所有进程的梯度。
3. 各个进程用平均后的梯度更新自己的参数，因为各个进程的初始参数、更新梯度一致，所以更新后的参数也完全相同。

## 基础概念

在 16 张显卡，16 的并行数下， DDP会同时启动 16 个进程。

`group` 进程组。默认情况下，只有一个组。

`world size` 全局的并行数。

`rank` 表示当前进程的序号，用于进程间通信。对于 16 的 world size 来说，就是 0, 1, 2, ... , 15。rank = 0  的进程就是 master 进程。

`local_rank` 表示每台机器上的进程的序号。机器一上有 0, 1, 2, ... , 7；机器二上也有 0, 1, 2, ... , 7。

## 命令行

```bash
python -m torch.distributed.launch --nproc_per_node=NUM_GPUS_YOU_HAVE train.py
```

其中 torch.distributed.launch 表示以分布式的方式启动训练，—nproc_per_node 指定一共多少节点，可以设置成显卡的个数。

**启动之后每个进程可以自动获取到参数**

local_rank 表示的是进程的优先级，该优先级是自动分配的。*不需要赋值，启动命令 torch.distributed.launch会自动赋值。*

world size 表示的一共运行的进程数，和 nproc_per_node 设置的数值相对应。

## 示例

```python
################
## main.py文件
import argparse
from tqdm import tqdm
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
# 新增：
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

### 1. 基础模块 ### 
# 假设我们的模型是这个，与DDP无关
class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
# 假设我们的数据是这个
def get_dataset():
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    my_trainset = torchvision.datasets.CIFAR10(root='./data', train=True, 
        download=True, transform=transform)
    # DDP：使用DistributedSampler，DDP帮我们把细节都封装起来了。
    #      用，就完事儿！sampler的原理，第二篇中有介绍。
    train_sampler = torch.utils.data.distributed.DistributedSampler(my_trainset)
    # DDP：需要注意的是，这里的batch_size指的是每个进程下的batch_size。
    #      也就是说，总batch_size是这里的batch_size再乘以并行数(world_size)。
    trainloader = torch.utils.data.DataLoader(my_trainset, 
        batch_size=16, num_workers=2, sampler=train_sampler)
    return trainloader
    
### 2. 初始化我们的模型、数据、各种配置  ####
# DDP：从外部得到local_rank参数
parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", default=-1, type=int)
FLAGS = parser.parse_args()
local_rank = FLAGS.local_rank

# DDP：DDP backend初始化
torch.cuda.set_device(local_rank)
dist.init_process_group(backend='nccl')  # nccl是GPU设备上最快、最推荐的后端

# 准备数据，要在DDP初始化之后进行
trainloader = get_dataset()

# 构造模型
model = ToyModel().to(local_rank)
# DDP: Load模型要在构造DDP模型之前，且只需要在master上加载就行了。
ckpt_path = None
if dist.get_rank() == 0 and ckpt_path is not None:
    model.load_state_dict(torch.load(ckpt_path))
# DDP: 构造DDP model
model = DDP(model, device_ids=[local_rank], output_device=local_rank)

# DDP: 要在构造DDP model之后，才能用model初始化optimizer。
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# 假设我们的loss是这个
loss_func = nn.CrossEntropyLoss().to(local_rank)

### 3. 网络训练  ###
model.train()
iterator = tqdm(range(100))
for epoch in iterator:
    # DDP：设置sampler的epoch，
    # DistributedSampler需要这个来指定shuffle方式，
    # 通过维持各个进程之间的相同随机数种子使不同进程能获得同样的shuffle效果。
    trainloader.sampler.set_epoch(epoch)
    # 后面这部分，则与原来完全一致了。
    for data, label in trainloader:
        data, label = data.to(local_rank), label.to(local_rank)
        optimizer.zero_grad()
        prediction = model(data)
        loss = loss_func(prediction, label)
        loss.backward()
        iterator.desc = "loss = %0.3f" % loss
        optimizer.step()
    # DDP:
    # 1. save模型的时候，和DP模式一样，有一个需要注意的点：保存的是model.module而不是model。
    #    因为model其实是DDP model，参数是被`model=DDP(model)`包起来的。
    # 2. 只需要在进程0上保存一次就行了，避免多次保存重复的东西。
    if dist.get_rank() == 0:
        torch.save(model.module.state_dict(), "%d.ckpt" % epoch)

################
## Bash运行
# DDP: 使用torch.distributed.launch启动DDP模式
# 使用CUDA_VISIBLE_DEVICES，来决定使用哪些GPU
# CUDA_VISIBLE_DEVICES="0,1" python -m torch.distributed.launch --nproc_per_node 2 main.py
```

新增：

```python
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", default=-1, type=int)
FLAGS = parser.parse_args()
local_rank = FLAGS.local_rank
```

```python
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
```

```python
torch.cuda.set_device(local_rank)
dist.init_process_group(backend='nccl')
```

```python
model.to(local_rank)
model = DDP(model, device_ids=[local_rank], output_device=[local_rank])
```

```python
train_dataloader.sampler.set_epoch(epoch)
```

```python
model.module.forword()
```

```python
CUDA_VISIBLE_DEVICES="0,1" python -m torch.distributed.launch --nproc_per_node 2 main.py
```

