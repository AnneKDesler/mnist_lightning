import argparse
import sys

import torch
import click

from model import MyAwesomeModel
from torch import nn, optim
import matplotlib.pyplot as plt
import numpy as np

import wandb
from omegaconf import OmegaConf
import pytorch_lightning as pl

def train(cfg):
    print("Training day and night")

    # TODO: Implement training loop here
    
    lr = cfg.lr
    batch_size = cfg.batch_size
    epochs = cfg.epochs

    train_set = torch.load("data/processed/train.pt")
    test_set = torch.load("data/processed/test.pt")

    trainloader = torch.utils.data.DataLoader(train_set, num_workers=8, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_set, num_workers=8, batch_size=batch_size)

    model = MyAwesomeModel(lr)
    #trainer = pl.Trainer(limit_train_batches=0.2, max_epochs=epochs, default_root_dir = 'models', logger=pl.loggers.WandbLogger(project="mnist"))
    trainer = pl.Trainer(max_epochs=epochs, default_root_dir = 'models', logger=pl.loggers.WandbLogger(project="mnist"), log_every_n_steps = 1)
    trainer.fit(model=model, train_dataloaders=trainloader, val_dataloaders=testloader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml', help = 'config file path')
    args =parser.parse_args()
    cfg = OmegaConf.load(args.config)
    train(cfg)
