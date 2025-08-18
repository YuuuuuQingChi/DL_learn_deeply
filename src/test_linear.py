# part1
# import math
# import time
# import numpy as np
# import torch
# from d2l import torch as d2l

# n = 10000
# a = torch.ones(n)
# b = torch.ones(n)
# c = torch.zeros(n)

# timer = d2l.Timer()
# for i in range(n):
#     c[i] = a[i] + b[i]
# print(f'{timer.stop():.5f} sec')
# timer.start()
# c = a + b
# print(f'{timer.stop():.5f} sec')
# import torch
# import math
# from d2l import torch as d21
# def normal(x,mu,sigma):
#     p = 1/math.sqrt(2*math.pi*sigma**2)
#     return p*torch.exp(-(1*(x - mu)**2)/(2*sigma**2))

# x = torch.arange(-7,7,0.001)

# params = [(0, 1), (0, 2), (3, 1)]
# d21.plt.plot(x,normal(x,3,1))
# d21.plt.show()

## part2
import random
import torch
import torch.utils.data as data
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from d2l import torch as d2l
import numpy as np
from PIL import Image, ImageDraw  # 添加Image模块导入
import test_linear_module as module


def synthetic_data(w, b, num_examples):
    """生成y=Xw+b+噪声"""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))


# 生成数据
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 10000)

# 数据加载
dataset_ = data.TensorDataset(features, labels)
dataloader_ = data.DataLoader(
    dataset=dataset_, batch_size=64, shuffle=True, drop_last=True
)

# 初始化
writer = SummaryWriter("runs")
module_ = module.linear()
loss_func = nn.MSELoss()
optimization = torch.optim.SGD(module_.parameters(), lr=0.001)

# 用于保存最佳模型
best_loss = float("inf")
best_model_state = None
epoch_step = 20
total_train_step = 0

for epoch in range(epoch_step):
    module_.train()
    epoch_loss = 0
    for data in dataloader_:
        x, y = data
        output = module_(x)
        output_loss = loss_func(output, y)

        optimization.zero_grad()
        output_loss.backward()
        optimization.step()

        epoch_loss += output_loss.item()
        total_train_step += 1
        writer.add_scalar("train_loss", output_loss.item(), total_train_step)

    avg_epoch_loss = epoch_loss / len(dataloader_)
    if avg_epoch_loss < best_loss:
        best_loss = avg_epoch_loss
        best_model_state = module_.state_dict().copy()
        torch.save(best_model_state, "best_model.pth")

    print(f"Epoch [{epoch+1}/{epoch_step}], Loss: {avg_epoch_loss:.4f}")

# 加载最佳模型
module_.load_state_dict(torch.load("best_model.pth"))

with torch.no_grad():
    # 获取训练好的参数
    w_trained = module_.sequential_[0].weight.data.squeeze()
    b_trained = module_.sequential_[0].bias.data.item()

    # 生成拟合直线数据 (取20个点)
    x_line = torch.linspace(-3, 3, 20)
    y_line = w_trained[0] * x_line + b_trained
    
    # 随机选择50个样本点用于可视化
    indices = np.random.choice(len(features), 50, replace=False)
    sample_x = features[indices, 0]  # 只取第一个特征
    sample_y = labels[indices]
    
    # 方法1：分别绘制散点和直线
    # 绘制原始数据点(散点)
    for i, (x, y) in enumerate(zip(sample_x, sample_y)):
        writer.add_scalar('Scatter_Plot/Data_Points', y.item(), i)
    
    # 绘制拟合直线
    for x, y in zip(x_line, y_line):
        writer.add_scalar('Line_Plot/Fitted_Line', y.item(), x.item())
    
    # 方法2：使用tags创建叠加效果
    for x, y in zip(sample_x, sample_y):
        writer.add_scalar('Data_and_Fit/Data', y.item(), x.item())
    
    for x, y in zip(x_line, y_line):
        writer.add_scalar('Data_and_Fit/Line', y.item(), x.item())



writer.close()