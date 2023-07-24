from os import listdir
from os.path import isdir, isfile, join
import re

import numpy as np
import pandas as pd
from numpy import genfromtxt
from pyapnea.oscar.oscar_constants import ChannelID
from pyapnea.oscar.oscar_getter import event_data_to_dataframe
from pyapnea.oscar.oscar_loader import load_session
from torch.utils.data import Dataset

from src.data.preparation_tasks import generate_annotations


# TODO : need sliding window : https://discuss.pytorch.org/t/is-there-a-data-datasets-way-to-use-a-sliding-window-over-time-series-data/115702/4
class RawBioPointDataset(Dataset):

    def __init__(self, input_type='csv', output_type='numpy', limits=None, channels=None):
        """

        :param output_type: 'numpy' or 'dataframe'
        """
        self.output_type = output_type
        data_path_biopoint = '../data/raw_biopoint/'
        self.list_files = [{'label': f, 'value': f, 'fullpath': join(data_path_biopoint, f)} for f in
                           listdir(data_path_biopoint)
                           if isdir(join(data_path_biopoint, f))]

        if limits is not None:
            self.list_files = self.list_files[:limits]

        self.channels = channels

    def __len__(self):
        return len(self.list_files)

    # https://medium.com/analytics-vidhya/optimized-ways-to-read-large-csvs-in-python-ab2b36a7914e
    def __getitem__(self, idx):
        def get_channel_from_filename(filename: str):
            filename = filename.replace('.', '_')
            elements = filename.split('_')
            return elements[-2]


        result = None
        fullpath = self.list_files[idx]['fullpath']
        files = [join(fullpath, f) for f in listdir(fullpath) if isfile(join(fullpath, f))]
        # TODO move to a specific boipoint util file
        global_df = pd.DataFrame(columns=['no_channel'])
        for f in files:
            channel = get_channel_from_filename(f)
            if (channel is None) or (channel in self.channels):
                print(channel)

                if self.output_type == 'dataframe':
                    try:
                        chunk = pd.read_csv(f, chunksize=1000000)
                        df = pd.concat(chunk)
                        if global_df.empty:
                            global_df = df
                        else:
                            global_df = pd.merge(global_df, df, on='Time', how='outer', suffixes=('', '_DROP')).filter(regex='^(?!.*_DROP)')
                    except pd.errors.EmptyDataError:
                        df = pd.DataFrame()

        if self.output_type == 'numpy':
            result = None
        if self.output_type == 'dataframe':
            result = global_df
        return result
