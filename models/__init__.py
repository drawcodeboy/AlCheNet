from .BaseNet.basenet import BaseNet
from .GraphNet.graphnet import GraphNet
from .Transformer.transformer import Transformer

def load_model(cfg):
    if cfg['name'] == 'BaseNet':
        return BaseNet.from_config(cfg)
    elif cfg['name'] == 'GraphNet':
        return GraphNet.from_config(cfg)
    elif cfg['name'] == 'Transformer':
        return Transformer.from_config(cfg)