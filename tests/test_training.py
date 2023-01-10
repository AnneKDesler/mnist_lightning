import torch
from src.models.model import MyAwesomeModel
def test_training():

    train_set = torch.load("data/processed/train.pt")
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=10)

    batch = next(iter(trainloader))
    model = MyAwesomeModel()
    
    images, labels = batch
    output = model(images)
    loss = model.criterion(output, labels)

    assert loss.item()>=0
