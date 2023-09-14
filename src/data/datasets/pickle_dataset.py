import pickle
import os

import numpy as np
from torch.utils.data import Dataset


class PickleDataset(Dataset):

    def __init__(self, output_type='numpy', limits=None, src_data_path=None):
        """
        :param output_type: 'numpy'. 'dataframe' is not allowed
        :param limits: slice to filter the dataset
        """
        self.output_type = 'numpy'
        input_file_inputs = os.path.join(src_data_path, 'inputs.pkl')
        input_file_gt = os.path.join(src_data_path, 'gt.pkl')
        with open(input_file_inputs, 'rb') as f_input:
            self.inputs = pickle.load(f_input)
        with open(input_file_gt, 'rb') as f_gt:
            self.ground_truths = pickle.load(f_gt)

        if limits is not None:
            self.inputs = self.inputs[limits]
            self.ground_truths = self.ground_truths[limits]

    def __len__(self):
        return len(self.ground_truths)

    def __getitem__(self, idx):
        result = np.float32(self.inputs[idx]), np.float32(self.ground_truths[idx])
        return result
