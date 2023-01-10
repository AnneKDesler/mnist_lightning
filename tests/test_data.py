import torch
from tests import _PATH_DATA
import os.path
import pytest

#@pytest.mark.skipif(not os.path.exists(_PATH_DATA + "/processed/train.pt") or not os.path.exists(_PATH_DATA + "/processed/test.pt"), reason="Data files not found")

def test_data():
    train_set = torch.load(_PATH_DATA + "/processed/train.pt")
    test_set = torch.load(_PATH_DATA + "/processed/test.pt")
    N_train = 25000
    N_test = 5000
    trainloader = torch.utils.data.DataLoader(train_set)
    testloader = torch.utils.data.DataLoader(test_set)

    assert len(train_set) == N_train, 'Train data is not the right length'
    assert len(test_set) == N_test, 'Test data is not the right length'

    unique = []
    for image, label in trainloader:
        if len(unique)==10:
            break
        assert image.shape == torch.Size([1, 1, 28, 28]), 'Train images are not the right shape'
        if label not in unique:
            unique.append(label.item)

    for i in range(10):
        assert i not in unique, 'Missing labels in train set'

    unique = []
    for image, label in testloader:
        if len(unique)==10:
            break
        assert image.shape == torch.Size([1, 1, 28, 28]), 'Test images are not the right shape'
        if label not in unique:
            unique.append(label.item)

    for i in range(10):
        assert i not in unique, 'Missing labels in test set'
    #assert next(iter(trainloader))[0].shape == [1, 1, 28, 28]
    #or training and N_test for test
    #assert that each datapoint has shape [1,28,28] or [728] depending on how you choose to format
    #assert that all labels are represented