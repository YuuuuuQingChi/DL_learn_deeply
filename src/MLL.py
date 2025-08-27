import numpy as np
import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l

d2l.DATA_HUB["kaggle_house_train"] = (
    d2l.DATA_URL + "kaggle_house_pred_train.csv",
    "585e9cc93e70b39160e7921475f9bcd7d31219ce",
)

d2l.DATA_HUB["kaggle_house_test"] = (
    d2l.DATA_URL + "kaggle_house_pred_test.csv",
    "fa19780a7b011d9b009e8bff8e99922a8ee2eb90",
)

train_data = pd.read_csv(d2l.download("kaggle_house_train"))
test_data = pd.read_csv(d2l.download("kaggle_house_test"))
print(train_data.shape)
print(test_data.shape)
print(train_data.iloc[0:4, [0, 1, -4, -1]])


features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:-1]))
print(features.iloc[0:4, [0, 1, -4, -1]])
index_numerical_features = features.dtypes[features.dtypes != "object"].index
# 获得所有数字的索引

# 记住dtype默认会去找所以列的数据类型
# 此外object'object' 类型通常表示字符串或分类数据
# 例如：[True, True, False]（前两列是数值型，第三列是字符串）

features[index_numerical_features] = features[index_numerical_features].apply(
    lambda x: ((x - x.mean()) / x.std())
)
print(features.iloc[0:4, [0, 1, -4, -1]])
# 这个是标准正态化
features[index_numerical_features] = features[index_numerical_features].fillna(0)

features = pd.get_dummies(features, dummy_na=True)

# 接下来，我们处理离散值。 这包括诸如“MSZoning”之类的特征。 我们用独热编码替换它们， 方法与前面将多类别标签转换为向量的方式相同 （请参见 3.4.1节）。 例如，“MSZoning”包含值“RL”和“Rm”。 我们将创建两个新的指示器特征“MSZoning_RL”和“MSZoning_RM”，其值为0或1。 根据独热编码，如果“MSZoning”的原始值为“RL”， 则：“MSZoning_RL”为1，“MSZoning_RM”为0。 pandas软件包会自动为我们实现这一点。
# pd.get_dummies()可以为分类变量转换为独热编码
n_train = train_data.shape[0]

train_features = torch.tensor(features[:n_train].values, dtype=torch.float32)
test_features = torch.tensor(features[n_train:-1].values, dtype=torch.float32)
train_label = torch.tensor(
    train_data.SalePrice.values.reshape(-1, 1), dtype=torch.float32
)

loss = nn.MSELoss()
in_features = train_features.shape[1]


def get_net():
    net = nn.Sequential(nn.Linear(in_features, 1))
    return net


def log_rmse(net, features, labels):
    # 为了在取对数时进一步稳定该值，将小于1的值设置为1
    clipped_preds = torch.clamp(net(features), 1, float("inf"))
    rmse = torch.sqrt(loss(torch.log(clipped_preds), torch.log(labels)))
    return rmse.item()
