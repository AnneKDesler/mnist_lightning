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
    
    train_sets = []
    for i in range(5):
        train = np.load("data/processed/" + f"train_{i}.npz")
        train_sets.append(torch.utils.data.TensorDataset(torch.tensor(train["images"]).float(), torch.tensor(train["labels"])))
        
    
    test = np.load("data/processed/test.npz")
    
    lr = cfg.lr
    batch_size = cfg.batch_size
    epochs = cfg.epochs

    test_set = torch.utils.data.TensorDataset(torch.tensor(test["images"]).float(), torch.tensor(test["labels"]))
    train_set = torch.utils.data.ConcatDataset(train_sets)
    #trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    #testloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)
    trainloader = torch.utils.data.DataLoader(train_set)
    testloader = torch.utils.data.DataLoader(test_set)
    

    model = MyAwesomeModel(lr)
    trainer = pl.Trainer(limit_train_batches=100, max_epochs=epochs, default_root_dir = 'models')
    trainer.fit(model=model, train_dataloaders=trainloader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml', help = 'config file path')
    args =parser.parse_args()
    cfg = OmegaConf.load(args.config)
    train(cfg)
