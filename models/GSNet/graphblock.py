import torch
from torch import nn
from torch_geometric.nn import SAGEConv, TopKPooling, global_mean_pool
from torch_geometric.data import Data, Batch

from ..mlp import MLP

class GraphBlock(nn.Module):
    def __init__(self,
                 in_channels:int,
                 hidden_channels:int,
                 out_channels:int,
                 n_layer:int,
                 mlp_hidden_features:int,
                 mlp_class_num:int):
        super().__init__()
        
        self.act_fn = nn.ReLU()
        self.sage_layers = nn.ModuleList([])
        self.sage_layers.append(SAGEConv(in_channels=in_channels,
                                         out_channels=hidden_channels))
        for layer in range(n_layer-2):
            self.sage_layers.append(SAGEConv(in_channels=hidden_channels,
                                             out_channels=hidden_channels))
        self.sage_layers.append(SAGEConv(in_channels=hidden_channels,
                                         out_channels=out_channels))
        
        self.pool = TopKPooling(in_channels=out_channels)
        
        self.mlp = MLP(in_features=out_channels,
                       hidden_features=mlp_hidden_features,
                       class_num=mlp_class_num)
    
    def forward(self, x):
        batch = self.create_batch_graph(x)
        
        for idx, layer in enumerate(self.sage_layers):
            x = layer(batch.x, batch.edge_index)
            if idx != (len(self.sage_layers)-1):
                x = self.act_fn(x)
            batch.x = x
                

        x, edge_index, edge_weight, batch_batch, _, _ = self.pool(
            x, batch.edge_index, edge_attr=batch.edge_weight, batch=batch.batch
        )

        x = self.mlp(global_mean_pool(x, batch_batch))
            
        return x
    
    def create_batch_graph(self, x):
        edge_index = x['edge_index']
        edge_weight = x['edge_weight']
        x = x['freq']
        
        B, N, D = x.shape
        data_list = []
        for b in range(B):
            data = Data(
                x=x[b],
                edge_index=edge_index[b],
                edge_weight=edge_weight[b]
            )
            data_list.append(data)
        
        batch = Batch.from_data_list(data_list)
        return batch