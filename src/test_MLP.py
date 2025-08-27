# part1

# import torch
# from torch import nn
# from d2l import torch as d2l
# net = nn.Sequential(nn.Flatten(),
#                     nn.Linear(784, 256),
#                     nn.ReLU(),
#                     nn.Linear(256, 10))

# def init(m):
#     if type(m) == nn.Linear:
#         nn.init.normal(m.weight,std = 0.01)

# net.apply(init);
# batch_size, lr, num_epochs = 256, 0.1, 10
# loss = nn.CrossEntropyLoss(reduction='none')
# trainer = torch.optim.SGD(net.parameters(), lr=lr)

# train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
# d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)

# d2l.plt.show()

# part2
# import torch
# import numpy as np
# import math
# from torch import nn as nn
# from d2l import torch as d2l

# max_degree = 20
# n_train, n_test = 100, 100

# true_W = np.zeros(max_degree)
# true_W[0:4] = [5, -1.2, -3.4, 5.6]

# test = np.random.normal(size=(n_train + n_test, 1))
# np.random.shuffle(test)
# ploy_features = np.power(
#     test, np.arange(max_degree).reshape(1, -1)
# )  # 这个power也可以广播操作
# for i in np.arange(max_degree):
#     ploy_features[:, i] /= math.gamma(i + 1)  # gamma(n)=(n-1)!
# labels = np.dot(ploy_features, true_W)
# labels += np.random.normal(scale=0.1, size=labels.shape)

# true_W, test, ploy_features, labels = [
#     torch.tensor(x, dtype=torch.float32) for x in [true_W, test, ploy_features, labels]
# ]


# def train(train_features, test_features, train_labels, test_labels, num_epochs=400):
#     loss = nn.MSELoss(reduction="none")
#     input_shape = train_features.shape[-1]  # 这个shape填入参数可以返回指定维度的大小
#     net = nn.Sequential(
#         nn.Linear(input_shape, 1, bias=False)
#     )  # 在初始化参数的时候，已经涵盖了
#     trainer = torch.optim.SGD(net.parameters(), lr=0.01)
#     batch_size = min(10, train_labels.shape[0])
#     train_iter = d2l.load_array(
#         (train_features, train_labels.reshape(-1, 1)), batch_size
#     )
#     test_iter = d2l.load_array(
#         (test_features, test_labels.reshape(-1, 1)), batch_size, is_train=False
#     )

#     animator = d2l.Animator(
#         xlabel="epoch",
#         ylabel="loss",
#         yscale="log",
#         xlim=[1, num_epochs],
#         ylim=[1e-3, 1e2],
#         legend=["train", "test"],
#     )
#     # 创建动画可视化器，用于绘制训练和测试损失曲线x轴: 训练轮数 y轴: 损失值(对数尺度) 显示训练和测试两条曲线
#     for epoch in range(num_epochs):
#         d2l.train_epoch_ch3(net, train_iter, loss, trainer)
#         if epoch == 0 or (epoch + 1) % 20 == 0:
#             animator.add(
#                 epoch + 1,
#                 (
#                     d2l.evaluate_loss(net, train_iter, loss),
#                     d2l.evaluate_loss(net, test_iter, loss),
#                 ),
#             )
#     print('weight:', net[0].weight.data.numpy())

# # 从多项式特征中选择前4个维度，即1,x,x^2/2!,x^3/3!
# train(
#     ploy_features[:n_train, :6],
#     ploy_features[n_train:, :6],
#     labels[:n_train],
#     labels[n_train:],
# )
# d2l.plt.show()


# part3

# import torch
# from torch import nn
# from d2l import torch as d2l
# n_train, n_test, num_inputs, batch_size = 20, 100, 200, 5
# true_w, true_b = torch.ones((num_inputs, 1)) * 0.01, 0.05
# train_data = d2l.synthetic_data(true_w, true_b, n_train)
# train_iter = d2l.load_array(train_data, batch_size)
# test_data = d2l.synthetic_data(true_w, true_b, n_test)
# test_iter = d2l.load_array(test_data, batch_size, is_train=False)
# def train_concise(wd):
#     net = nn.Sequential(nn.Linear(num_inputs, 1))
#     for param in net.parameters():
#         param.data.normal_()
#     loss = nn.MSELoss(reduction="none")
#     num_epochs, lr = 100, 0.003
#     # 偏置参数没有衰减
#     trainer = torch.optim.SGD(
#         [{"params": net[0].weight, "weight_decay": wd}, {"params": net[0].bias}], lr=lr
#     )
#     animator = d2l.Animator(
#         xlabel="epochs",
#         ylabel="loss",
#         yscale="log",
#         xlim=[5, num_epochs],
#         legend=["train", "test"],
#     )
#     for epoch in range(num_epochs):
#         for X, y in train_iter:
#             trainer.zero_grad()
#             l = loss(net(X), y)
#             l.mean().backward()
#             trainer.step()
#         if (epoch + 1) % 5 == 0:
#             animator.add(
#                 epoch + 1,
#                 (
#                     d2l.evaluate_loss(net, train_iter, loss),
#                     d2l.evaluate_loss(net, test_iter, loss),
#                 ),
#             )
#     print("w的L2范数：", net[0].weight.norm().item())
#     d2l.plt.show()

# train_concise(0.0)

# part4
# import torch
# from torch import nn
# from d2l import torch as d2l

# dropout1, dropout2 = 0.2, 0.5
# lr = 0.1
# net = nn.Sequential(
#     nn.Flatten(),
#     nn.Linear(784, 256),
#     nn.ReLU(),
#     # 在第一个全连接层之后添加一个dropout层
#     nn.Dropout(dropout1),
#     nn.Linear(256, 256),
#     nn.ReLU(),
#     # 在第二个全连接层之后添加一个dropout层
#     nn.Dropout(dropout2),
#     nn.Linear(256, 10),
# )


# def init_weights(m):
#     if type(m) == nn.Linear:
#         nn.init.normal_(m.weight, std=0.01)


# net.apply(init_weights)
# trainer = torch.optim.SGD(net.parameters(), lr=lr)
# num_epochs, lr, batch_size = 10, 0.5, 256
# loss = nn.CrossEntropyLoss(reduction='none')
# train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
# trainer = torch.optim.SGD(net.parameters(), lr=lr)
# d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
# d2l.plt.show()
