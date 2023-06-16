from os import listdir
from os.path import isfile, join

from torch.utils.data import Dataset
import numpy as np
import pandas as pd

from pyapnea.oscar.oscar_loader import load_session
from pyapnea.oscar.oscar_getter import event_data_to_dataframe
from pyapnea.oscar.oscar_constants import ChannelID


# TODO : need sliding window : https://discuss.pytorch.org/t/is-there-a-data-datasets-way-to-use-a-sliding-window-over-time-series-data/115702/4
class OscarDataset(Dataset):

    def __init__(self):
        data_path_cpap1 = '../data/raw/ResMed_23192565579/Events'
        data_path_cpap2 = '../data/raw/ResMed_23221085377/Events'
        self.list_files = [{'label': f, 'value': f, 'fullpath': join(data_path_cpap1, f)} for f in
                           listdir(data_path_cpap1)
                           if isfile(join(data_path_cpap1, f))]
        self.list_files.extend(
            [{'label': f, 'value': f, 'fullpath': join(data_path_cpap2, f)} for f in listdir(data_path_cpap2) if
             isfile(join(data_path_cpap2, f))])

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, idx):
        oscar_session_data = load_session(self.list_files[idx]['fullpath'])
        df = event_data_to_dataframe(oscar_session_data, [ChannelID.CPAP_FlowRate.value,
                                                          ChannelID.CPAP_Obstructive.value])

        if 'Obstructive' not in df.columns:
            df['Obstructive'] = 0.0

        df_annotation = df[['time_utc', 'Obstructive']].copy()
        df_annotation['Obstructive'] = df_annotation['Obstructive'].apply(lambda x: 1 if not pd.isnull(x) else np.nan)
        df_annotation['Obstructive'].fillna(0, inplace=True)

        # assert len(df['FlowRate'] == len(df_annotation['Obstructive']))
        return df[['FlowRate']].to_numpy(), df_annotation[['Obstructive']].to_numpy()
