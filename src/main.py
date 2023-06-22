import os
from os import listdir
from os.path import join
from functools import partial

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

import torch.optim as optim

from poutyne.framework import Model

from src.data.datasets.raw_oscar_dataset import RawOscarDataset
from src.data.datasets.processed_dataset import ProcessedDataset
from src.models.basic_lstm_model import BasicLSTMModel
from src.data.ts_collator import TSCollator
from src.metrics.accuracy import accuracy
from src.data.preparation_tasks import generate_rolling_window_dataframes

from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = "cpu"
print(device)

batch_size = 16

# oscar_dataset = RawOscarDataset(output_type='dataframe')
# processed_dataset = ProcessedDataset(output_type='numpy')


# windowed_dfs = []
# with tqdm(total=len(oscar_dataset), position=0, leave=False, colour='red', ncols=80) as pbar:
#     for idx_ts, ts in enumerate(oscar_dataset):
#         dfs = generate_rolling_window_dataframes(ts, 500, keep_last_incomplete=False)
#         os.mkdir('../data/processing/windowed_dataset/feather/'+'df_'+str(idx_ts))
#         with tqdm(total=len(dfs), position=1, leave=False, colour='green', ncols=80) as pbar2:
#             for idx_df, df in enumerate(dfs):
#                 df_name = 'df_'+str(idx_ts)+'_'+str(idx_df)
#                 #df.to_hdf('../data/processing/window_dataset_'+str(idx_ts)+'.h5', df_name, mode='a')
#                 df.reset_index(inplace=True)
#                 df.to_feather('../data/processing/windowed_dataset/feather/'+'df_'+str(idx_ts)+'/' + df_name + '.feather')
#                 pbar2.update(1)
#         pbar.update(1)



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