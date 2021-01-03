import torch

import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, drop_prob=0.5):
        super(CNN, self).__init__()
        self.conv2d1 = nn.Conv2d(1, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2d2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv2d3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv2d4 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.max_pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(p=drop_prob)
        self.num_out = 3*3*128
        self.fc1 = nn.Linear(self.num_out, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 128)
        self.fc4 = nn.Linear(128, 7)
    def forward(self, x):
        out = self.bn1(self.max_pool(F.relu(self.conv2d1(x))))
        out = self.bn2(self.max_pool(F.relu(self.conv2d2(out))))
        out = self.bn3(self.max_pool(F.relu(self.conv2d3(out))))
        out = self.bn4(self.max_pool(F.relu(self.conv2d4(out)))) 

        out = out.view(-1, self.num_out)

        out = F.relu(self.dropout(self.fc1(out)))
        out = F.relu(self.dropout(self.fc2(out)))
        out = F.relu(self.dropout(self.fc3(out)))
        out = self.fc4(out)
        return out

