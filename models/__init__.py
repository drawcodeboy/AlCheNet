from .BaseNet.basenet import BaseNet
from .GCNet.gcnet import GCNet
from .ChebNet.chebnet import ChebNet
from .GSNet.gsnet import GSNet
from .GATNet.gatnet import GATNet
from .Transformer.transformer import Transformer

def load_model(cfg):
    if cfg['name'] == 'BaseNet':
        return BaseNet.from_config(cfg)
    elif cfg['name'] == 'GCNet':
        return GCNet.from_config(cfg)
    elif cfg['name'] == 'ChebNet':
        return ChebNet.from_config(cfg)
    elif cfg['name'] == 'GSNet':
        return GSNet.from_config(cfg)
    elif cfg['name'] == 'GATNet':
        return GATNet.from_config(cfg)
    elif cfg['name'] == 'Transformer':
        return Transformer.from_config(cfg)