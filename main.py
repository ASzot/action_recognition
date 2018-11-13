from dataset import MomentsDataset
from torch.utils.data import Dataset, DataLoader

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class CnnNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 29 * 29, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 29 * 29)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



dataset_path = '/hdd/datasets/moments/Moments_in_Time_256x256_30fps/'

ds = MomentsDataset(dataset_path)

#label_count = ds.get_num_classes()
batch_size = 4

dataloader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=1)

model = CnnNet(ds.get_num_classes())
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

N_EPOCHS = 10

for epoch_i in range(N_EPOCHS):
    for batch in dataloader:

        X = batch['images']
        labels = batch['label']

        #X = X.reshape(batch_size, 3, 128, 128, 90)
        X = X.reshape(batch_size, 3, 128, 128)
        X = X.float()
        predicted_labels = model(X)

        labels = labels.long()

        loss = criterion(predicted_labels, labels)

        loss.backward()
        optimizer.step()

        print('Loss: %.5f' % (loss.data.cpu().numpy()))




