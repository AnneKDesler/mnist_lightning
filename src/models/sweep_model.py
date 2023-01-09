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

def train(config=None):
    print("Training day and night")

    # Initialize a new wandb run
    with wandb.init(config=config):
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config

        # TODO: Implement training loop here
        model = MyAwesomeModel()
        train_sets = []
        for i in range(5):
            train = np.load("data/processed/" + f"train_{i}.npz")
            train_sets.append(torch.utils.data.TensorDataset(torch.tensor(train["images"]).float(), torch.tensor(train["labels"])))
            
        
        test = np.load("data/processed/" + "test.npz")
        
        lr =config.lr
        batch_size = config.batch_size
        epochs = config.epochs

        test_set = torch.utils.data.TensorDataset(torch.tensor(test["images"]).float(), torch.tensor(test["labels"]))
        train_set = torch.utils.data.ConcatDataset(train_sets)
        trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
        testloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)

        
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.NLLLoss()
        steps = 0
        running_loss = 0
        print_every = 60
        losses_train = []
        losses_test = []
        accuracys = []


        for e in range(epochs):
            # Model in training mode, dropout is on
            model.train()
            print("Epoch: {}/{}.. ".format(e+1, epochs))

            for images, labels in trainloader:
                steps += 1
                optimizer.zero_grad()
                output = model.forward(images)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
                
                losses_train.append(loss.item())
                wandb.log({"train loss": loss.item()})
            accuracy = 0
            loss = 0
            for images, labels in testloader:
                
                output = model(images)
                loss_i = criterion(output, labels).item()
                loss += loss_i

                ## Calculating the accuracy 
                # Model's output is log-softmax, take exponential to get the probabilities
                ps = torch.exp(output)
                # Class with highest probability is our predicted class, compare with true label
                equality = (labels.data == ps.max(1)[1])
                # Accuracy is number of correct predictions divided by all predictions, just take the mean
                accuracy_i = equality.type_as(torch.FloatTensor()).mean()
                accuracy += accuracy_i
                wandb.log({"test loss": loss_i})
                wandb.log({"accuracy": accuracy_i})

            losses_test.append(loss/len(testloader))
            accuracys.append(accuracy/len(testloader))


if __name__ == "__main__":
    #parser = argparse.ArgumentParser()
    #parser.add_argument('--sweep_config', type=str, default='sweep.yaml', help = 'sweep config file path')
    #args =parser.parse_args()
    #sweep_config = args.sweep_config
    #sweep_config = OmegaConf.load(args.sweep_config)
    sweep_config = {
    'method': 'random',
    'name': 'sweep',
    'metric': {
        'goal': 'minimize', 
        'name': 'test loss'
		},
    'parameters': {
        'batch_size': {'values': [500,1000,1500]},
        'epochs': {'values': [10]},
        'lr': {'max': 0.1, 'min': 0.0001}
     }
}
    sweep_id = wandb.sweep(sweep=sweep_config, project="mnist")
    wandb.agent(sweep_id, train, count=5)

