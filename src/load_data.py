from os import listdir
from os.path import join
from functools import partial

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

import torch.optim as optim

from poutyne.framework import Model

from src.data.oscar_dataset import OscarDataset
from src.models.basic_lstm_model import BasicLSTMModel
from src.data.ts_collator import TSCollator
from src.metrics.accuracy import accuracy

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"
print(device)

batch_size = 1

oscar_dataset = OscarDataset()

col = TSCollator()

# https://www.scottcondron.com/jupyter/visualisation/audio/2020/12/02/dataloaders-samplers-collate.html
train_loader = DataLoader(dataset=oscar_dataset, batch_size=batch_size, shuffle=False, collate_fn=col.collate_batch)
valid_loader = DataLoader(dataset=oscar_dataset, batch_size=batch_size, shuffle=False, collate_fn=col.collate_batch)

model = BasicLSTMModel()
optimizer = optim.Adam(model.parameters())
loss_fn = nn.CrossEntropyLoss()

accuracy_fn = partial(accuracy, device=device)

model = Model(model, optimizer, loss_fn,  batch_metrics=['accuracy'],  device=device)

model.fit_generator(train_loader, valid_loader, epochs=1)

# TODO test https://www.kaggle.com/code/omershect/learning-pytorch-lstm-deep-learning-with-m5-data