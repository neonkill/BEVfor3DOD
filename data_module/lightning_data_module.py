import torch
import pytorch_lightning as pl


class DataModule(pl.LightningDataModule):

    def __init__(self, data_cfg, loader_cfg):
        super().__init__()

        self.data_cfg = data_cfg
        self.loader_cfg = loader_cfg


    def get_loader(self, split, shuffle):
        
        version = 'v1.0-' + split
        loader_cfg = dict(self.loader_cfg)

        dataset = NuimagesDataset(tf_cfg=self.data_cfg, version=version)

        return torch.utils.data.DataLoader(dataset, 
                                        shuffle=shuffle, 
                                        **loader_cfg)


    def train_dataloader(self, shuffle=True):
        return self.get_loader('train', shuffle=shuffle)

    def val_dataloader(self, shuffle=True):
        return self.get_loader('val', shuffle=shuffle)

    def eval_dataloader(self, shuffle=False):
        return self.get_loader('eval', shuffle=shuffle)