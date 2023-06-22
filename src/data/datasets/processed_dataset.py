from os import listdir, walk
from os.path import isfile, join
from glob import glob

from torch.utils.data import Dataset
import numpy as np
import pandas as pd

from pyapnea.oscar.oscar_loader import load_session
from pyapnea.oscar.oscar_getter import event_data_to_dataframe
from pyapnea.oscar.oscar_constants import ChannelID


class ProcessedDataset(Dataset):

    def __init__(self, output_type = 'numpy'):
        """

        :param output_type: 'numpy' or 'dataframe'
        """
        self.output_type = output_type
        data_path = '../data/processing/windowed_dataset/2023-06-21-500pts-dsv1.0/feather/'
        list_files = [y for x in walk(data_path) for y in glob(join(x[0], '*.feather'))]
        self.list_files = [{'label': f, 'value': f, 'fullpath': f} for f in list_files]

        self.list_files = self.list_files[:1000]

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, idx):
        result = None
        df = pd.read_feather(self.list_files[idx]['fullpath'])
        if self.output_type == 'numpy':
            result = df[['FlowRate']].to_numpy(), df[['Obstructive']].to_numpy()
        if self.output_type == 'dataframe':
            df.set_index('time_utc', inplace=True)
            df.drop('index', inplace=True)
            result = df
        return result
