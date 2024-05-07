from glob import glob
from os import walk
import os.path
import pickle
import zipfile
import tempfile

import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class ProcessedDataset(Dataset):

    def __init__(self, data_path, getitem_type='numpy', limits=None):
        """
        :param getitem_type: 'numpy' or 'dataframe'
        :param limits: slice to filter the dataset
        """
        self.getitem_type = getitem_type
        self.data_path = data_path
        self.real_data_path = data_path

        files = [f for f in os.listdir(self.data_path) if os.path.isfile(os.path.join(self.data_path, f))]
        if len(files) == 1 and os.path.splitext(files[0])[1] == '.zip':
            print('Unzipping {}'.format(files[0]))
            self.temp_dir = tempfile.TemporaryDirectory()
            with zipfile.ZipFile(os.path.join(self.data_path, files[0]), 'r') as zip_ref:
                zip_ref.extractall(self.temp_dir.name)
            self.real_data_path = self.temp_dir.name
            print('Unzipping {} done'.format(files[0]))

        if os.path.isfile(os.path.join(self.real_data_path, 'inputs.pkl')):
            self.file_format = 'pickle'
            input_file_inputs = os.path.join(self.real_data_path, 'inputs.pkl')
            input_file_gt = os.path.join(self.real_data_path, 'gt.pkl')
            with open(input_file_inputs, 'rb') as f_input:
                self.inputs = pickle.load(f_input)
            with open(input_file_gt, 'rb') as f_gt:
                self.ground_truths = pickle.load(f_gt)
        else:
            self.file_format = 'feather'
            list_files = [y for x in walk(self.real_data_path) for y in glob(os.path.join(x[0], '*.feather'))]
            self.list_files = [{'label': f, 'value': f, 'fullpath': f} for f in list_files]

        if limits is not None:
            if self.file_format == 'feather':
                self.list_files = self.list_files[limits]
            if self.file_format == 'pickle':
                self.inputs = self.inputs[limits]
                self.ground_truths = self.ground_truths[limits]

    def __len__(self):
        result = 0
        if self.file_format == 'feather':
            result = len(self.list_files)
        if self.file_format == 'pickle':
            result = len(self.ground_truths)
        return result

    def __getitem__(self, idx):
        result = None
        if self.file_format == 'feather':
            df = pd.read_feather(self.list_files[idx]['fullpath'])
            assert df['time_utc'].is_monotonic_increasing

            if self.getitem_type == 'numpy':
                result = df[['FlowRate']].to_numpy(), df[['ApneaEvent']].to_numpy()
            if self.getitem_type == 'dataframe':
                df.set_index('time_utc', inplace=True)
                df.drop('index', axis=1, errors='ignore', inplace=True)
                result = df
        if self.file_format == 'pickle':
            result = np.float32(self.inputs[idx]), np.float32(self.ground_truths[idx])
        return result
