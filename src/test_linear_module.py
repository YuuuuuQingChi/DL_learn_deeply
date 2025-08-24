import torch

class linear(torch.nn.Module):
    def __init__(self):
        super(linear, self).__init__()
        self.sequential_ = torch.nn.Sequential(torch.nn.Linear(2,1))

    def forward(self,x):
        return self.sequential_(x)

class softmax_(torch.nn.Module):
    def __init__(self):
        super(softmax_, self).__init__()
        self.sequential_ = torch.nn.Sequential(torch.nn.Flatten(),torch.nn.Linear(in_features=784,out_features=10))

    def forward(self,x):
        return self.sequential_(x)