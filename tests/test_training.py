import torch
from src.models.model import MyAwesomeModel
from tests import _PATH_DATA
import pytest
import os

@pytest.mark.skipif(not os.path.exists(_PATH_DATA + "/processed/train.pt"), reason="Data files not found")
def test_training():

    train_set = torch.load(_PATH_DATA + "/processed/train.pt")
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=10)

    batch = next(iter(trainloader))
    model = MyAwesomeModel()
    
    images, labels = batch
    output = model(images)
    loss = model.criterion(output, labels)

    assert loss.item()>=0
