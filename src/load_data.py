from os import listdir
from os.path import isfile, join

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.utils.data as data
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

from pyapnea.oscar.oscar_loader import load_session
from pyapnea.oscar.oscar_getter import event_data_to_dataframe, get_channel_from_code
from pyapnea.oscar.oscar_constants import CHANNELS, ChannelID

# TODO : need sliding window : https://discuss.pytorch.org/t/is-there-a-data-datasets-way-to-use-a-sliding-window-over-time-series-data/115702/4
class OscarDataset(Dataset):

    def __init__(self):
        data_path_cpap1 = '../data/raw/ResMed_23192565579/Events'
        data_path_cpap2 = '../data/raw/ResMed_23221085377/Events'
        self.list_files = [{'label': f, 'value': f, 'fullpath': join(data_path_cpap1, f)} for f in listdir(data_path_cpap1)
                      if isfile(join(data_path_cpap1, f))]
        self.list_files.extend(
            [{'label': f, 'value': f, 'fullpath': join(data_path_cpap2, f)} for f in listdir(data_path_cpap2) if
             isfile(join(data_path_cpap2, f))])

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, idx):
        oscar_session_data = load_session(self.list_files[idx]['fullpath'])
        df = event_data_to_dataframe(oscar_session_data, [ChannelID.CPAP_FlowRate.value,
                                                          ChannelID.CPAP_ClearAirway.value,
                                                          ChannelID.CPAP_Obstructive.value])

        if 'Obstructive' not in df.columns:
            df['Obstructive'] = 0.0

        df_annotation = df[['time_utc', 'Obstructive']].copy()
        df_annotation['Obstructive'] = df_annotation['Obstructive'].apply(lambda x:1 if not pd.isnull(x) else np.nan)
        df_annotation['Obstructive'].fillna(0, inplace=True)

        assert len(df['FlowRate']==len(df_annotation['Obstructive']))
        return df[['FlowRate']].to_numpy(), df_annotation[['Obstructive']].to_numpy()


# https://github.com/iid-ulaval/CCF-dataset/blob/main/experiment/src/data/collator/embedding_collator.py
class EmbeddingCollator:
    def __init__(self,
                 padding_value: int = 0,
                 ):
        self.padding_value = padding_value

    def collate_batch(
        self, batch):
        input_tensor, target_tensor, lengths = zip(*[(
            torch.Tensor(embeded_sequence), torch.Tensor(target),
            len(embeded_sequence)
        ) for embeded_sequence, target in batch])

        input_tensor = pad_sequence(input_tensor,
                                    batch_first=True,
                                    padding_value=self.padding_value)

        target_tensor = pad_sequence(target_tensor,
                                     batch_first=True,
                                     padding_value=self.padding_value)

        lengths_tensor = torch.Tensor(lengths)

        return ((input_tensor, lengths_tensor), target_tensor)

# https://machinelearningmastery.com/lstm-for-time-series-prediction-in-pytorch/
class BasicLSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=50, num_layers=1, batch_first=True)
        self.linear = nn.Linear(50, 1)
    def forward(self, input_, lengths):
        lstm_output, _ = self.lstm(input_)

        output = self.linear(lstm_output)

        return output

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

oscar_dataset = OscarDataset()

batch_size = 2

col = EmbeddingCollator()

# https://www.scottcondron.com/jupyter/visualisation/audio/2020/12/02/dataloaders-samplers-collate.html
train_loader = DataLoader(dataset=oscar_dataset, batch_size=batch_size, shuffle=False, collate_fn=col.collate_batch)
valie_loader = DataLoader(dataset=oscar_dataset, batch_size=batch_size, shuffle=False, collate_fn=col.collate_batch)

# for inputData, target in enumerate(train_loader):
#      print('input', inputData)
#      print('target', target[0][0].shape, target[0][1], target[1].shape)

model = BasicLSTMModel()
optimizer = optim.Adam(model.parameters())
loss_fn = nn.MSELoss()

n_epochs = 10
for epoch in range(n_epochs):
    print('epoque', epoch)
    model.train()
    for ((X_batch, X_lengths), y_batch) in train_loader:
        y_pred = model(X_batch, X_lengths)
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # Validation
    if epoch % 100 != 0:
        continue
    # model.eval()
    # with torch.no_grad():
    #     y_pred = model(X_train)
    #     train_rmse = np.sqrt(loss_fn(y_pred, y_train))
    #     y_pred = model(X_test)
    #     test_rmse = np.sqrt(loss_fn(y_pred, y_test))
    #  print("Epoch %d: train RMSE %.4f, test RMSE %.4f" % (epoch, train_rmse, test_rmse))