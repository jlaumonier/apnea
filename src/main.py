import os
from os import listdir
from os.path import join
from functools import partial

import torch
import hydra
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

import torch.optim as optim

from poutyne.framework import Model

from src.data.datasets.raw_oscar_dataset import RawOscarDataset
from src.data.datasets.processed_dataset import ProcessedDataset
from src.models.basic_lstm_model import BasicLSTMModel
from src.data.ts_collator import TSCollator
from src.metrics.accuracy import accuracy
from src.data.preparation_tasks import generate_rolling_window_dataframes, generate_all_rolling_window



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = "cpu"
print(device)

batch_size = 16

@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(conf):
    oscar_dataset = RawOscarDataset(output_type='dataframe', limits=1)
    generate_all_rolling_window(oscar_dataset=oscar_dataset,
                                length=500,
                                keep_last_incomplete=False,
                                output_dir_path='../data/processing/windowed/')

# processed_dataset = ProcessedDataset(output_type='numpy')

# col = TSCollator()
#
# # https://www.scottcondron.com/jupyter/visualisation/audio/2020/12/02/dataloaders-samplers-collate.html
# train_loader = DataLoader(dataset=processed_dataset, batch_size=batch_size, shuffle=True, collate_fn=col.collate_batch)
# valid_loader = DataLoader(dataset=processed_dataset, batch_size=batch_size, shuffle=True, collate_fn=col.collate_batch)
#
# model = BasicLSTMModel()
# optimizer = optim.Adam(model.parameters())
# loss_fn = nn.CrossEntropyLoss()
#
# accuracy_fn = partial(accuracy, device=device)
#
# model = Model(model, optimizer, loss_fn,  batch_metrics=['accuracy'],  device=device)
#
# model.fit_generator(train_loader, valid_loader, epochs=10)

# TODO test https://www.kaggle.com/code/omershect/learning-pytorch-lstm-deep-learning-with-m5-data


if __name__ == "__main__":
    main()
