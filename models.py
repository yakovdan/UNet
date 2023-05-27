import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(3)
        self.conv1 = nn.Conv2d(3, 16, 3, padding="same")
        self.relu1 = nn.ReLU()

        self.bn1a = nn.BatchNorm2d(3)
        self.conv1a = nn.Conv2d(3, 16, 5, padding="same")
        self.relu1a = nn.ReLU()

        self.bn1b = nn.BatchNorm2d(3)
        self.conv1b = nn.Conv2d(3, 16, 7, padding="same")
        self.relu1b = nn.ReLU()

        self.bn1c = nn.BatchNorm2d(3)
        self.conv1c = nn.Conv2d(3, 16, 9, padding="same")
        self.relu1c = nn.ReLU()

        self.bn2 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 32, 3, padding="same")
        self.relu2 = nn.ReLU()

        self.bn3 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, 3, padding="same")
        self.relu3 = nn.ReLU()

        self.bn4 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 32, 5,  padding="same")
        self.relu4 = nn.ReLU()

        self.bn5 = nn.BatchNorm2d(32)
        self.conv5 = nn.Conv2d(32, 32, 5, padding="same")
        self.relu5 = nn.ReLU()

        self.fc1 = nn.Linear(64 * 32 * 32, 1000)
        self.fc2 = nn.Linear(1000, 250)
        self.fc3 = nn.Linear(250, 100)

        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.dropout3 = nn.Dropout(0.25)
        self.dropout4 = nn.Dropout(0.25)
        self.dropout5 = nn.Dropout(0.25)
        self.dropout6 = nn.Dropout(0.25)
        self.dropout7 = nn.Dropout(0.25)
        self.dropout8 = nn.Dropout(0.25)

        self.init_weights()


    def init_weights(self):
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv1a.weight)
        nn.init.xavier_uniform_(self.conv1b.weight)
        nn.init.xavier_uniform_(self.conv1c.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.conv3.weight)
        nn.init.xavier_uniform_(self.conv4.weight)
        nn.init.xavier_uniform_(self.conv5.weight)

    def forward(self, x):
        x1 = self.relu1(self.conv1(self.bn1(x)))
        x2 = self.relu1a(self.conv1a(self.bn1a(x)))
        x3 = self.relu1b(self.conv1b(self.bn1b(x)))
        x4 = self.relu1c(self.conv1c(self.bn1c(x)))
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.dropout1(x)
        x = self.relu2(self.conv2(self.bn2(x)))
        x = self.dropout2(x)
        x = self.relu3(self.conv3(self.bn3(x)))
        x = self.dropout3(x)
        x = self.relu4(self.conv4(self.bn4(x)))
        x = self.dropout4(x)
        y = self.relu5(self.conv5(self.bn5(x)))
        y = self.dropout5(y)

        x = torch.cat((x, y), dim=1)
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.dropout6(x)
        x = F.relu(self.fc2(x))
        x = self.dropout7(x)
        x = self.fc3(x)

        return x
