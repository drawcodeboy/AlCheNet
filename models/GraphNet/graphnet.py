import torch
from torch import nn
from torch_geometric.nn import GCN, TopKPooling

import sys, os
sys.path.append(os.getcwd())
from models.mlp import MLP
from ..ConvNet.convnet import ConvNet

class GraphNet(nn.Module):
    def __init__(self,
                 convnet_channels:int,
                 convnet_window_size:int,
                 convnet_seq_len:int,
                 convnet_node_dim:int,
                 in_channels:int,
                 hidden_channels:int,
                 out_channels:int,
                 n_layer:int,
                 mlp_hidden_features:int,
                 mlp_class_num:int):
        super().__init__()
        
        self.convnet = ConvNet(channels=convnet_channels,
                               window_size=convnet_window_size,
                               seq_len=convnet_seq_len,
                               node_dim=convnet_node_dim)
        
        self.gcn = GCN(in_channels=in_channels,
                       hidden_channels=hidden_channels,
                       out_channels=out_channels,
                       num_layers=n_layer,
                       act='relu')
        
        self.pool = TopKPooling(in_channels=out_channels)
        
        self.mlp = MLP(in_features=out_channels,
                       hidden_features=mlp_hidden_features,
                       class_num=mlp_class_num)
    
    def forward(self, x):
        edge_index = x['edge_index']
        edge_weight = x['edge_weight']
        x = x['freq']
        
        x = self.convnet(x)
        
        x = self.gcn(x, edge_index, edge_weight=edge_weight)
        print("pass")
        
        x = self.pool(x)
        
        x = self.mlp(x)
        
        return x
    
    @classmethod
    def from_config(cls, cfg):
        return cls(convnet_channels=cfg['convnet_channels'],
                   convnet_window_size=cfg['convnet_window_size'],
                   convnet_seq_len=cfg['convnet_seq_len'],
                   convnet_node_dim=cfg['convnet_node_dim'],
                   in_channels=cfg['in_channels'],
                   hidden_channels=cfg['hidden_channels'],
                   out_channels=cfg['out_channels'],
                   n_layer=cfg['n_layer'],
                   mlp_hidden_features=cfg['mlp_hidden_features'],
                   mlp_class_num=cfg['mlp_class_num'])
        
if __name__ == '__main__':
    model = GraphNet(32, 64, 128, 6, 48, 3)
    
    