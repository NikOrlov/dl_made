import torch.nn as nn
import torch.nn.functional as F
from src.config import LABELS


class Model(nn.Module):
    def __init__(self, in_channels):
        super(Model, self).__init__(),
        self.conv1 = nn.Conv2d(in_channels, 64, 3)
        self.mp1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(64, 128, 3)
        self.mp2 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(128, 256, 3)
        self.conv4 = nn.Conv2d(256, 256, 3, padding=1)
        self.mp3 = nn.MaxPool2d((1, 2), stride=2)

        self.conv5 = nn.Conv2d(256, 512, 3)
        self.bn1 = nn.BatchNorm2d(512)

        self.conv6 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(512)
        self.mp4 = nn.MaxPool2d((1, 2), stride=2)

        self.conv7 = nn.Conv2d(512, 512, 2)

        self.rnn = nn.LSTM(
            input_size=512,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = nn.Linear(512, len(LABELS))

    def forward(self, x):
        bs = x.shape[0]
        x = F.relu(self.conv1(x))
        x = self.mp1(x)

        x = F.relu(self.conv2(x))
        x = self.mp2(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.mp3(x)

        x = F.relu(self.bn1((self.conv5(x))))
        x = F.relu(self.bn2((self.conv6(x))))
        x = self.mp4(x)

        x = F.relu(self.conv7(x))

        x = x.reshape(bs, -1, 5).permute(0, 2, 1)

        x = F.relu(self.rnn(x)[0])
        x = self.fc(x)

        return x.permute(0, 2, 1)
