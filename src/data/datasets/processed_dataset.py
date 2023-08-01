from glob import glob
from os import walk
from os.path import join

import pandas as pd
from torch.utils.data import Dataset


class ProcessedDataset(Dataset):

    def __init__(self, output_type='numpy', limits=None):
        """
        :param output_type: 'numpy' or 'dataframe'
        :param limits: slice to filter the dataset
        """
        self.output_type = output_type
        data_path = '../data/processing/windowed/feather/'
        list_files = [y for x in walk(data_path) for y in glob(join(x[0], '*.feather'))]
        self.list_files = [{'label': f, 'value': f, 'fullpath': f} for f in list_files]

        if limits is not None:
            self.list_files = self.list_files[limits]

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, idx):
        result = None
        df = pd.read_feather(self.list_files[idx]['fullpath'])

        if self.output_type == 'numpy':
            result = df[['FlowRate']].to_numpy(), df[['ApneaEvent']].to_numpy()
        if self.output_type == 'dataframe':
            df.set_index('time_utc', inplace=True)
            df.drop('index', axis=1, errors='ignore', inplace=True)
            result = df
        return result
