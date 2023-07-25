from os import listdir
from os.path import isdir, isfile, join
import datetime as dt

import numpy as np
import pytz
import pandas as pd
from numpy import genfromtxt
from pyapnea.oscar.oscar_constants import ChannelID
from pyapnea.oscar.oscar_getter import event_data_to_dataframe
from pyapnea.oscar.oscar_loader import load_session
from torch.utils.data import Dataset

from src.data.preparation_tasks import generate_annotations


# TODO : need sliding window : https://discuss.pytorch.org/t/is-there-a-data-datasets-way-to-use-a-sliding-window-over-time-series-data/115702/4
class RawBioPointDataset(Dataset):

    ORIG_FREC = {'ECG': 500,
             'EMG': 2000,
             'BioImp': 40,
             'IMU': 50,
             'TEMPERATURE': 1}

    def __init__(self, input_type='csv', output_type='numpy', limits=None, channels=None, resampling_freq=None):
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

        self.dest_resampling_freq = resampling_freq

    def __len__(self):
        return len(self.list_files)

    # https://medium.com/analytics-vidhya/optimized-ways-to-read-large-csvs-in-python-ab2b36a7914e
    def __getitem__(self, idx):
        def get_info_from_filename(filename: str):
            filename = filename.replace('.', '_')
            elements = filename.split('_')
            start_time = elements[-6]+'.'+elements[-5]+'.'+elements[-4]+' '+elements[-3]
            return elements[-2], start_time

        result = None
        fullpath = self.list_files[idx]['fullpath']
        files = [{'fullpath': join(fullpath, f), 'filename':f} for f in listdir(fullpath) if isfile(join(fullpath, f))]
        # TODO move to a specific boipoint util file
        global_df = pd.DataFrame(columns=['no_channel'])
        for f in files:
            channel, start_time = get_info_from_filename(f['filename'])
            start_time_timestamp = dt.datetime.strptime(start_time, "%Y.%m.%d %Hh%Mm%S")
            tz = pytz.timezone("Canada/Eastern")
            start_time_timestamp = tz.localize(start_time_timestamp, is_dst=None)
            start_time_timestamp_utc = start_time_timestamp.astimezone(pytz.utc)
            if (channel is None) or (channel in self.channels):
                print(channel, start_time_timestamp)

                if self.output_type == 'dataframe':
                    try:
                        chunk = pd.read_csv(f['fullpath'], chunksize=1000000)
                        df = pd.concat(chunk)

                        # recalculate the time column
                        df['time_utc'] = (pd.date_range(start=start_time_timestamp_utc,
                                                        periods=len(df),
                                                        freq=str(1 / self.ORIG_FREC[channel]) + 'S'))

                        df.set_index('time_utc', inplace=True)
                        df.sort_index(inplace=True)

                        # resampling
                        if self.dest_resampling_freq:
                            df = df.resample(str(1 / self.dest_resampling_freq) + 'S').mean()

                        if global_df.empty:
                            global_df = df
                        else:
                            global_df = pd.merge(global_df, df, left_index=True, right_index=True, how='outer', suffixes=('', '_DROP')).filter(regex='^(?!.*_DROP)')
                    except pd.errors.EmptyDataError:
                        df = pd.DataFrame()

        global_df.sort_index(inplace=True)
        global_df.drop('Time', axis=1, inplace=True)
        global_df.fillna(method='ffill', inplace=True)

        if self.output_type == 'numpy':
            result = None
        if self.output_type == 'dataframe':
            result = global_df
        return result
