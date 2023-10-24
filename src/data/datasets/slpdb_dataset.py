import os.path
from typing import List, Optional
from os import listdir
from os.path import isfile, join, isdir
from enum import Enum, auto

import pandas as pd
from torch.utils.data import Dataset
import wfdb
import numpy as np

class AnnotID(Enum):
    SLPDB_AWAKE = 'W'
    SLPDB_SLEEP_STG_1 = '1'
    SLPDB_SLEEP_STG_2 = '2'
    SLPDB_SLEEP_STG_3 = '3'
    SLPDB_SLEEP_STG_4 = '4'
    SLPDB_REM_SLEEP_STG_4 = 'R'
    SLPDB_HYPOPNEA = 'H'
    SLPDB_HYPOPNEA_AROUSAL = 'HA'
    SLPDB_OBSTRUCTIVE = 'OA'
    SLPDB_OBSTRUCTIVE_AROUSAL = 'X'
    SLPDB_CENTRAL = 'CA'
    SLPDB_CENTRAL_AROUSAL = 'CAA'
    SLPDB_LEG = 'L'
    SLPDB_LEG_AROUSAL = 'LA'
    SLPDB_UNSPECIFIED_AROUSAL = 'A'
    SLPDB_MVT_TIME = 'M'

def generate_annotations_slpdb(df: pd.DataFrame, length_event=None, output_events_merge=None):
    """
    Generate annotations from a dataframe containing annotation at one point only.
    Note : Not sure if this is the best way to generate annotations because of the 30s after the point of apnea annotation.
    The real events cannot be identified.

    :param df: source dataframe used to generate annotation.
    :param length_event: length of the events to complete annotations. format in Offset aliases. \
     None for keeping annotation as-is (default).
    :param output_events_merge: list of AnnotID (not value of AnnotID) to merge to become the ApneaEvent. None for all apnea events
    """
    result = df.copy()
    if output_events_merge:
        possible_apnea_events = output_events_merge
    else:
        possible_apnea_events = [AnnotID.SLPDB_HYPOPNEA.value,
                                 AnnotID.SLPDB_HYPOPNEA_AROUSAL.value,
                                 AnnotID.SLPDB_OBSTRUCTIVE.value,
                                 AnnotID.SLPDB_OBSTRUCTIVE_AROUSAL.value,
                                 AnnotID.SLPDB_CENTRAL.value,
                                 AnnotID.SLPDB_CENTRAL_AROUSAL.value]
    possible_apnea_events_str = '|'.join(possible_apnea_events)
    events_in_origin = result['Event'].str.contains(possible_apnea_events_str)
    if len(events_in_origin) == 0:
        result['ApneaEvent'] = 0.0
    else:
        result['ApneaEvent'] = events_in_origin
        result['ApneaEvent'] = result['ApneaEvent'].astype(int)
        if length_event is not None:
            list_index_annot = result.index[result['ApneaEvent'] == 1].tolist()
            for annot_index in list_index_annot:
                indexes = result.index[((result.index >= annot_index) &
                                        (result.index <= (annot_index + pd.to_timedelta(length_event))))]
                result.loc[indexes, 'ApneaEvent'] = 1

    return result

class SLPDB_Dataset(Dataset):
    """
    download with 'wget -r -N -c -np https://physionet.org/files/slpdb/1.0.0/'
    """

    def __init__(self, data_path, output_type='numpy', limits=None, output_events_merged: Optional[List[AnnotID]] = None):
        """
        :param output_type: 'numpy' or 'dataframe'
        :param limits: determine the number of elements to get
        :param output_events_merged: List of apnea events (AnnotID) to merge into the 'ApneaEvent' column, None means all apnea event types are merged
        """
        self.output_type = output_type
        self.list_files = [join(data_path, f) for f in
                           listdir(data_path)
                           if isfile(join(data_path, f)) and os.path.splitext(f)[1]=='.dat']
        self.list_files = [w.replace(".dat", "") for w in self.list_files]

        if limits is not None:
            self.list_files = self.list_files[limits]

        self.output_events_merged = output_events_merged


    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, idx):
        result = None

        slpdb_header = wfdb.rdheader(self.list_files[idx])
        slpdb_record = wfdb.rdrecord(self.list_files[idx])
        slpdb_annotation = wfdb.rdann(self.list_files[idx], extension='st')
        df = slpdb_record.to_dataframe()

        df.reset_index(inplace=True)
        df_annot = pd.DataFrame(slpdb_annotation.aux_note, index = slpdb_annotation.sample)
        df['Event'] = df_annot
        df['Event'].fillna('', inplace=True)
        df.set_index('index', inplace=True)
        df.sort_index(inplace=True)

        if 'Resp (nasal)' in slpdb_header.sig_name:
            df.rename(columns={"Resp (nasal)":"FlowRate"}, inplace=True)
            idx_chan_resp = slpdb_header.sig_name.index('Resp (nasal)')
            gain_chan_resp = slpdb_header.adc_gain[idx_chan_resp]
            df['FlowRate'] = df['FlowRate'] * gain_chan_resp

            df = generate_annotations_slpdb(df, length_event='30S', output_events_merge=self.output_events_merged)
        else:
            df['FlowRate'] = 0.0
            df['ApneaEvent'] = 0

        if self.output_type == 'numpy':
            result = df[['FlowRate']].to_numpy(), df[['ApneaEvent']].to_numpy()
        if self.output_type == 'dataframe':
            result = df
        return result
