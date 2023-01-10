import torch
import pytest
from src.models.model import MyAwesomeModel

def test_model():
    model = MyAwesomeModel()
    image = torch.rand(1, 1, 28, 28)
    output = model(image)
    assert len(output) == 1

    image = torch.rand(10, 1, 28, 28)
    output = model(image)
    assert len(output) == 10
