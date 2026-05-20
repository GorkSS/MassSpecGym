import sys
#sys.path.append('../')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, \
    Dropout
from torch.optim import Adam, SGD
from copy import deepcopy

# Sources: 
# https://media.geeksforgeeks.org/wp-content/uploads/20250617160652239500/NNwith_early_stopping.ipynb
# https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html#save-load-state-dict-recommended

class EarlyStopping:
    def __init__(self, patience=5, delta=0, device = "cpu"):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_model_state = None,
        self.device = device

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.best_model_state = deepcopy(model.cpu().state_dict())
            model.to(self.device)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_state = deepcopy(model.cpu().state_dict())
            model.to(self.device)
            self.counter = 0


class ANN_Class(nn.Module):
    def __init__(self, input_dim=1000, output_dim=300):
        super(ANN_Class, self).__init__()

        self.fc1 = nn.Linear(input_dim, 2048)

        self.fc2 = nn.Linear(2048, 1024)

        self.fc3 = nn.Linear(1024, output_dim)

        self.dropout = nn.Dropout(0.25)


    def encode(self, x):

        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = x.view(x.size(0), -1)

        return x

    def forward(self, x):
        x = self.encode(x)
        return x
