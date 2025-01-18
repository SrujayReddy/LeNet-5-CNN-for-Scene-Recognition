# python imports
import os
from tqdm import tqdm

# torch imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# helper functions for computer vision
import torchvision
import torchvision.transforms as transforms


class LeNet(nn.Module):
    def __init__(self, input_shape=(32, 32), num_classes=100):
        super(LeNet, self).__init__()

        # Convolutional Layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)

        # Pooling Layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully Connected Layers
        self.fc1 = nn.Linear(in_features=16 * 5 * 5, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=num_classes)

    def forward(self, x):
        shape_dict = {}

        # Stage 1: Conv1 -> ReLU -> Pool
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        shape_dict[1] = list(x.shape)

        # Stage 2: Conv2 -> ReLU -> Pool
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        shape_dict[2] = list(x.shape)

        # Stage 3: Flatten
        x = torch.flatten(x, start_dim=1)  # Flatten all dimensions except batch
        shape_dict[3] = list(x.shape)

        # Stage 4: FC1 -> ReLU
        x = self.fc1(x)
        x = F.relu(x)
        shape_dict[4] = list(x.shape)

        # Stage 5: FC2 -> ReLU
        x = self.fc2(x)
        x = F.relu(x)
        shape_dict[5] = list(x.shape)

        # Stage 6: FC3 (Output Layer)
        out = self.fc3(x)
        shape_dict[6] = list(out.shape)

        return out, shape_dict


def count_model_params():
    '''
    Return the number of trainable parameters of LeNet in millions.
    '''
    lenet_model = LeNet()

    # Initialize a counter for trainable parameters
    trainable_param_count = 0

    # Iterate over all parameters in the model
    for param in lenet_model.parameters():
        if param.requires_grad:
            trainable_param_count += param.numel()

    # Convert the total number of parameters to millions
    trainable_params_millions = trainable_param_count / 1e6

    return trainable_params_millions


def train_model(model, train_loader, optimizer, criterion, epoch):
    """
    model (torch.nn.module): The model created to train
    train_loader (pytorch data loader): Training data loader
    optimizer (optimizer.*): A instance of some sort of optimizer, usually SGD
    criterion (nn.CrossEntropyLoss) : Loss function used to train the network
    epoch (int): Current epoch number
    """
    model.train()
    train_loss = 0.0
    for input, target in tqdm(train_loader, total=len(train_loader)):
        # 1) zero the parameter gradients
        optimizer.zero_grad()
        # 2) forward + backward + optimize
        output, _ = model(input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # Update the train_loss variable
        # .item() detaches the node from the computational graph
        train_loss += loss.item()

    train_loss /= len(train_loader)
    print('[Training set] Epoch: {:d}, Average loss: {:.4f}'.format(epoch+1, train_loss))

    return train_loss


def test_model(model, test_loader, epoch):
    model.eval()
    correct = 0
    with torch.no_grad():
        for input, target in test_loader:
            output, _ = model(input)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_acc = correct / len(test_loader.dataset)
    print('[Test set] Epoch: {:d}, Accuracy: {:.2f}%\n'.format(
        epoch+1, 100. * test_acc))

    return test_acc
