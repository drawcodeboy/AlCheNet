import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self,
                 in_features:int,
                 hidden_features:int,
                 class_num:int):
        super().__init__()

        self.li1 = nn.Linear(in_features, hidden_features)
        self.li2 = nn.Linear(hidden_features, hidden_features)
        self.final = nn.Linear(hidden_features, class_num)
        self.acti = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.2)
        
    def forward(self, x):
        x = self.acti(self.li1(x))
        x = self.dropout(x)
        x = self.acti(self.li2(x))
        x = self.dropout(x)
        x = self.final(x)
        return x