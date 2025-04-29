from .Mamba.mamba import Mamba
from .Transformer.transformer import Transformer
from .Classifier.mamba_classifier import MambaClassifier
from .Classifier.transformer_classifier import TransformerClassifier

def load_model(cfg):
    #if cfg['name'] == 'VAE':
    #    return VAE.from_config(cfg)
    
    if cfg['name'] == 'Mamba':
        return Mamba.from_config(cfg)
    
    elif cfg['name'] == 'Transformer':
        return Transformer.from_config(cfg)
    
    elif cfg['name'] == 'MambaClassifier':
        return MambaClassifier.from_config(cfg)
    
    elif cfg['name'] == 'TransformerClassifier':
        return TransformerClassifier.from_config(cfg)