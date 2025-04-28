from .nuro_dataset import NuroDataset

def load_dataset(cfg):
    if cfg['name'] == 'Nuro':
        return NuroDataset.from_config(cfg)
    else:
        raise Exception(f"Dataset: {cfg['name']} is not supported.")