from .Mamba.mamba import Mamba
from .Transformer.transformer import Transformer

def load_model(cfg):
    #if cfg['name'] == 'VAE':
    #    return VAE.from_config(cfg)
    
    if cfg['name'] == 'Mamba':
        return Mamba.from_config(cfg)
    
    elif cfg['name'] == 'Transformer':
        return Transformer.from_config(cfg)