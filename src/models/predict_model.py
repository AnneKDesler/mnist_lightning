import argparse
import sys

import torch
import click

from model import MyAwesomeModel
from torch import nn, optim
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf

def evaluate(cfg, checkpoint):
    print("Evaluating until hitting the ceiling")

    hyps = cfg.hyperparameters
    batch_size = hyps.batch_size

    # TODO: Implement evaluation logic here
    model = torch.load(checkpoint)
    test = np.load("data/processed/test.npz")
    test_set = torch.utils.data.TensorDataset(torch.tensor(test["images"]).float(), torch.tensor(test["labels"]))
    testloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)

    accuracy = 0
    for images, labels in testloader:
        
        output = model(images)

        ## Calculating the accuracy 
        # Model's output is log-softmax, take exponential to get the probabilities
        ps = torch.exp(output)
        # Class with highest probability is our predicted class, compare with true label
        equality = (labels.data == ps.max(1)[1])
        # Accuracy is number of correct predictions divided by all predictions, just take the mean
        accuracy += equality.type_as(torch.FloatTensor()).mean()

    accuracy /= len(testloader)
    print(accuracy.item())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml', help = 'config file path')
    parser.add_argument('--checkpoint', type=str, default='models/checkpoints/checkpoint.pth', help = 'checkpoint file path')
    args =parser.parse_args()
    cfg = OmegaConf.load(args.config)
    evaluate(cfg, args.checkpoint)
    
