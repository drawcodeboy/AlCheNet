from .Mamba.mamba import Mamba

def load_model(cfg):
    #if cfg['name'] == 'VAE':
    #    return VAE.from_config(cfg)
    
    if cfg['name'] == 'Mamba':
        return Mamba.from_config(cfg)