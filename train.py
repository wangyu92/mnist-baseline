import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model import Model

# Hyperparameters
BATCH_SIZE      = 64
NUM_EPOCH       = 2
LR              = 1e-3

def get_dataset():
    data_dir = './mnist_data/'

    apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

    train_dataset = datasets.MNIST(
        data_dir, train=True, download=True, transform=apply_transform
    )

    test_dataset = datasets.MNIST(
        data_dir, train=False, download=True, transform=apply_transform
    )

    return train_dataset, test_dataset

def train_step(model, x, y):
    optimizer = optim.Adam(model.parameters(), LR)
    optimizer.zero_grad()
    pred = model(x)
    loss = nn.CrossEntropyLoss()(pred, y)
    loss.backward()
    optimizer.step()
    return loss

def test_model(model, test_dataset):
    device = torch.device(next(model.parameters()).device if next(model.parameters()).is_cuda else 'cpu')

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    testloader = DataLoader(test_dataset, batch_size=128,
                            shuffle=False)

    for batch_idx, (x, y) in enumerate(testloader):
        x, y = x.to(device), y.to(device)

        # Inference
        outputs = model(x)
        batch_loss = nn.NLLLoss()(outputs, y)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, y)).item()
        total += len(y)

    accuracy = correct/total
    return accuracy, loss


if __name__ == '__main__':
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

    train_dataset, test_dataset = get_dataset()

    model = Model().to(device)

    print(test_model(model, test_dataset))

    trainloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    for _ in range(NUM_EPOCH):
        for i, (x, y) in enumerate(trainloader):
            x, y = x.to(device), y.to(device)
            loss = train_step(model, x, y)

            if i % 10 == 0:
                print(test_model(model, test_dataset))