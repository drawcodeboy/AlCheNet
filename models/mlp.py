import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self,
                 in_features:int,
                 hidden_features:int,
                 class_num:int):
        super().__init__()

        self.li1 = nn.Linear(in_features, hidden_features)
        self.norm = nn.BatchNorm1d(num_features=hidden_features)
        self.relu = nn.ReLU()
        
        self.li2 = nn.Linear(hidden_features, class_num)
        
    def forward(self, x):
        x = self.relu(self.norm(self.li1(x)))
        
        x = self.li2(x)
        return x