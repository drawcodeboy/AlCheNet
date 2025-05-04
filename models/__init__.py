from .ConvNet.convnet import ConvNet
from .GraphNet.graphnet import GraphNet

def load_model(cfg):
    if cfg['name'] == 'ConvNet':
        return ConvNet.from_config(cfg)
    elif cfg['name'] == 'GraphNet':
        return GraphNet.from_config(cfg)