import torch
from torch import nn, optim
import torch.nn.functional as F
import pytorch_lightning as pl
from omegaconf import OmegaConf
import argparse

class MyAwesomeModel(pl.LightningModule):
    def __init__(self, lr=1e-2):
        super().__init__()
        self.criterion = nn.NLLLoss()
        self.lr = lr
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=3,
			kernel_size=(5, 5))
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=5,
			kernel_size=(5, 5))
        self.fc1 = nn.Linear(in_features=2000, out_features=100)
        self.fc2 = nn.Linear(in_features=100, out_features=10)

    def forward(self, x):
        if x.shape[1] != 1 or x.shape[2] != 28 or x.shape[3] != 28:
            raise ValueError('Expected each sample to have shape [1, 28, 28]')
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))

        x = F.log_softmax(self.fc2(x), dim=1)

        return x
    
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        images, labels = batch
        output = self.forward(images)
        loss = self.criterion(output, labels)
        self.log("train_loss", loss)#, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        acc = (labels == output.argmax(dim=-1)).float().mean()
        self.log("train_acc", acc)#, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        output = self.forward(images)
        loss = self.criterion(output, labels)
        self.log("val_loss", loss)#, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        acc = (labels == output.argmax(dim=-1)).float().mean()
        self.log("val_acc", acc)#, on_step=True, on_epoch=True, prog_bar=True, logger=True)


    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml', help = 'config file path')
    args =parser.parse_args()
    cfg = OmegaConf.load(args.config)
    model = MyAwesomeModel(cfg.lr)
