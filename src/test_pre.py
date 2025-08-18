# part 1
# import torch

# x = torch.randn([3,3])
# y = torch.arange(1,10).reshape([3,3])
# z = torch.cat((x,y))
# print(z)
# z = torch.cat((x,y),dim= 1)
# print(z)

# part 2
# import os
# import pandas as pd
# import torch 
# import numpy
# os.makedirs(os.path.join(".", "pre_data"), exist_ok= True)
# datafile = os.path.join(".", "pre_data", "house_tiny.csv")
# with open(datafile, mode="w") as f:
#     f.write("NumRooms,Alley,Price\n")  # 列名
#     f.write("NA,Pave,127500\n")  # 每行表示一个数据样本
#     f.write("2,NA,106000\n")
#     f.write("4,NA,178100\n")
#     f.write("NA,NA,140000\n")

# data = pd.read_csv(datafile)

# inputs = data.iloc[:,0:2]
# ouuput = data.iloc[:,-1]
# print(inputs)
# inputs = inputs.fillna(inputs.mean())
# print(inputs)
# inputs = pd.get_dummies(inputs, dummy_na=True)
# print(inputs.shape)
# x = torch.tensor(inputs.to_numpy(),dtype=float)
# print(x)

# A = torch.arange(12).reshape(3,4)
# B = A.clone()
# print(A,id(B))
# sum = A.sum(axis = 1,keepdim=True)
# print(sum,sum.shape)
# A = A.float()
# A.sum(axis = 0)
# A.size(axis = 0)
# A.mean(axis = 0)


# A_reduce_axis = A / A.sum()
# print(A_reduce_axis)
# print(A.mean(axis = 0))

# A_cumsum = A.cumsum(dim=0,dtype=float)
# print(A_cumsum)

# x = torch.ones(16,dtype=float)
# y = torch.arange(0,16,dtype=float)
# print(x ,y ,torch.dot(x,y))

#part3


# import numpy as np
# from matplotlib_inline import backend_inline
# from d2l import torch as d21


# def f(x):
#     return 3 * x ** 2 - 4 * x

# def numerical_lim(f, x, h):
#     return (f(x + h) - f(x)) / h

# h = 0.1
# for i in range(5):
#     print(f'h={h:.5f}, numerical limit={numerical_lim(f, 1, h):.5f}')
#     h *= 0.1

# x = np.arange(0, 3, 0.1)
# d21.plot(x, [f(x), 2 * x - 3], 'x', 'f(x)', legend=['f(x)', 'Tangent line (x=1)'])

#part4

# import torch

# x = torch.arange(4.0,requires_grad=True)
# # print(x.grad)
# # y = 2* torch.dot(x,x)
# # #如果 x 是 1维向量（如 shape=[n]），直接使用 x 即可，无需转置（因为向量点积 torch.dot(x, x) 不需要转置）
# # y.backward()
# # print(y)


# y = x * x
# u = y.detach()
# z = u * x

# z.sum().backward()
# x.grad.zero_()
# x.grad == u

# def f(a):
#     b = a * 2
#     while b.norm() < 1000:
#         b = b * 2
#     if b.sum() > 0:
#         c = b
#     else:
#         c = 100 * b
#     return c

# a =torch.randn(size=(),requires_grad= True)
# d = f(a)
# d.backward()
# print(a.grad == d / a ,a)

##part5

import torch
from torch.distributions import multinomial
from d2l import torch as d2l

fair_probs = torch.ones([6]) / 6
# 将结果存储为32位浮点数以进行除法
counts = multinomial.Multinomial(1000, fair_probs).sample()
# 相对频率作为估计值

print(counts / 1000)
counts = multinomial.Multinomial(10, fair_probs).sample((500,))
print(counts )
cum_counts = counts.cumsum(dim=0)
estimates = cum_counts / cum_counts.sum(dim=1, keepdims=True)

d2l.set_figsize((6, 4.5))
for i in range(6):
    d2l.plt.plot(estimates[:, i].numpy(),
                 label=("P(die=" + str(i + 1) + ")"))
d2l.plt.axhline(y=0.167, color='black', linestyle='dashed')
d2l.plt.gca().set_xlabel('Groups of experiments')
d2l.plt.gca().set_ylabel('Estimated probability')
d2l.plt.legend();
d2l.plt.show()