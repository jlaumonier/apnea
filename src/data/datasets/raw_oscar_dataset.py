import os.path
from typing import List, Optional
from os import listdir
from os.path import isfile, join, isdir

from pyapnea.oscar.oscar_constants import ChannelID
from pyapnea.oscar.oscar_getter import event_data_to_dataframe
from pyapnea.oscar.oscar_loader import load_session
from torch.utils.data import Dataset

from src.data.preparation_tasks import generate_annotations


# TODO : need sliding window : https://discuss.pytorch.org/t/is-there-a-data-datasets-way-to-use-a-sliding-window-over-time-series-data/115702/4
class RawOscarDataset(Dataset):

    def __init__(self, data_path, output_type='numpy', limits=None,
                 output_events_merged: Optional[List[ChannelID]] = None,
                 channel_ids: Optional[List[ChannelID]] = None):
        """

        :param output_type: 'numpy' or 'dataframe'
        :param limits: determine the number of elements to get
        :param output_events_merged: List of apnea events (ChannelID) to merge into the 'ApneaEvent' column, None means all apnea event types are merged
        :param channel_ids: List of channel to get. If None, only CPAP_FlowRate is get.
        """
        self.output_type = output_type
        # l=listdir(data_path)
        list_machines = [d for d in listdir(data_path) if isdir(os.path.join(data_path, d))]
        data_path_cpap = [os.path.join(data_path, d, 'Events') for d in list_machines]
        # data_path_cpap1 = '../data/raw/ResMed_23192565579/Events'
        # data_path_cpap2 = '../data/raw/ResMed_23221085377/Events'
        data_path_cpap1 = data_path_cpap[0]
        self.list_files = [{'label': f, 'value': f, 'fullpath': join(data_path_cpap1, f)} for f in
                           listdir(data_path_cpap1)
                           if isfile(join(data_path_cpap1, f))]

        if len(data_path_cpap) == 2:
            data_path_cpap2 = data_path_cpap[1]

            self.list_files.extend(
                [{'label': f, 'value': f, 'fullpath': join(data_path_cpap2, f)} for f in listdir(data_path_cpap2) if
                 isfile(join(data_path_cpap2, f))])

        if limits is not None:
            self.list_files = self.list_files[:limits]

        self.list_files = sorted(self.list_files, key=lambda x: x['fullpath'])

        if channel_ids is not None:
            self.channel_ids = channel_ids
        else:
            self.channel_ids = [ChannelID.CPAP_FlowRate.value]

        self.output_events_merged = output_events_merged

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, idx):
        result = None
        oscar_session_data = load_session(self.list_files[idx]['fullpath'])
        channel_to_get = [ChannelID.CPAP_Obstructive.value,  # Apnée obstructive
                          ChannelID.CPAP_ClearAirway.value,  # Apnée centrale
                          ChannelID.CPAP_Hypopnea.value,  # Hypopnée
                          ChannelID.CPAP_Apnea.value,  # Non déterminé
                          ]
        channel_to_get.extend(self.channel_ids)
        df = event_data_to_dataframe(oscar_session_data,
                                     channel_ids=channel_to_get,
                                     mis_value_strategy={ChannelID.CPAP_FlowRate.value: 'ignore'})

        df.set_index('time_utc', inplace=True)
        df.sort_index(inplace=True)
        df = generate_annotations(df, length_event='10S', output_events_merge=self.output_events_merged)

        if self.output_type == 'numpy':
            result = df[['FlowRate']].to_numpy(), df[['ApneaEvent']].to_numpy()
        if self.output_type == 'dataframe':
            result = df
        return result
