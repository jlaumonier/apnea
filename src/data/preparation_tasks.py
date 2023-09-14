import os
from tqdm import tqdm
from p_tqdm import p_map
import pickle
import random

import pandas as pd
import numpy as np
from torch.utils.data import Dataset

from pyapnea.oscar.oscar_constants import CHANNELS, ChannelID
from src.data.datasets.pickle_dataset import PickleDataset
from .utils import get_nb_events


def align_channels(df: pd.DataFrame, reference_channel: str, period_ref_channel: str) -> pd.DataFrame:
    """
    Align channels on one timestamp of one reference channel
    Assuming the column 'time_utc' is used for the reference timestamp

    :param df: dataframe containing data to align.
    :param reference_channel: Name of the column representing the channel on which other channels must be aligned.
    :param period_ref_channel: period of the reference channel (pandas TimeDelta format)
    :return: dataframe where all channels are aligned
    """
    list_channels_to_align = [col for col in df.columns if col not in ['time_utc', reference_channel]]
    list_channels_for_merging = list_channels_to_align.copy()
    list_channels_for_merging.append('time_utc')

    df_annotation_to_merge = df[df[reference_channel].isnull()]

    df_annotation_to_merge = df_annotation_to_merge[list_channels_for_merging]
    df_non_annotation_or_not_merge = df[~df[reference_channel].isnull()]

    result_df = pd.merge_asof(df_non_annotation_or_not_merge, df_annotation_to_merge, on='time_utc',
                              direction='nearest',
                              allow_exact_matches=False,
                              tolerance=pd.Timedelta(period_ref_channel))

    for c in list_channels_to_align:
        result_df[c] = result_df[c + '_x'].combine_first(result_df[c + '_y'])
        result_df.drop([c + '_x', c + '_y'], axis=1, inplace=True)
    return result_df


# thanks aneroid - I don't understand that, in 2023, this function is not included in pandas !
# https://stackoverflow.com/questions/66482997/pandas-sliding-window-over-a-dataframe-column
def _sliding_window_iter(df, length, keep_last_incomplete):
    if len(df) % length == 0 or not keep_last_incomplete:
        # if there is no last incomplete or if we do not want to keep it
        max_range = len(df) - length + 1
    else:
        # there is last incomplete and we do want to keep it
        max_range = len(df) - length + 2
    for start_row in range(0, max_range, length):
        yield df[start_row:start_row + length]


def generate_rolling_window_dataframes(df: pd.DataFrame,
                                       length: int,
                                       keep_last_incomplete=True,
                                       sort_index = False) -> list[pd.DataFrame]:
    """
    This method generates subsets of the original dataset, with fixed length, using sliding window.
    Does not support overlap yet.
    :param df: original dataframe
    :param length: length of all the subsets in points
    :param keep_last_incomplete: True to keep the last incomplete slice (with len<lentgth) if it exists
    :return: a list of dataframes containing each window
    """
    if sort_index:
        sorted_df = df.sort_index()
    else:
        sorted_df = df
    result = [d for d in _sliding_window_iter(sorted_df, length, keep_last_incomplete)]
    return result


def generate_all_rolling_window(oscar_dataset: Dataset,
                                output_dir_path: str,
                                length: int,
                                keep_last_incomplete=True,
                                output_format='feather') -> None:

    def  _process_element(idx_ts, ts):
        dfs = generate_rolling_window_dataframes(ts, length=length, keep_last_incomplete=keep_last_incomplete)
        output_dir = os.path.join(output_dir_path, output_format, 'df_' + str(idx_ts))
        os.makedirs(output_dir, exist_ok=True)
        for idx_df, df in enumerate(dfs):
            df_name = 'df_' + str(idx_ts) + '_' + str(idx_df)
            # df.to_hdf('../data/processing/window_dataset_'+str(idx_ts)+'.h5', df_name, mode='a')
            df.reset_index(inplace=True)
            df.to_feather(os.path.join(output_dir, df_name + '.feather'))

    # for idx_ts, ts in tqdm(enumerate(oscar_dataset), total=len(oscar_dataset)):
    #     _process_element(idx_ts, ts)
    p_map(_process_element, range(len(oscar_dataset)), oscar_dataset)


def generate_pickle_dataset(oscar_dataset: Dataset,
                            output_dir_path: str) -> None:
    """
    This method generates a piclke numpy dataset from windows feather dataset set as numpy
    :param oscar_dataset: processed dataset with numpy output
    :param output_dir_path: path of the pickle outputs
    """

    os.makedirs(output_dir_path, exist_ok=True)
    output_file_inputs = os.path.join(output_dir_path, 'inputs.pkl')
    output_file_gt = os.path.join(output_dir_path, 'gt.pkl')
    inputs = []
    ground_truths = []

    for input, gt in tqdm(oscar_dataset):
        inputs.append(input)
        ground_truths.append(gt)

    inputs = np.array(inputs)
    ground_truths = np.array(ground_truths)

    with open(output_file_inputs, 'wb') as f_input:
        pickle.dump(inputs, f_input)
    with open(output_file_gt, 'wb') as f_gt:
        pickle.dump(ground_truths, f_gt)


def generate_overfitting_dataset(oscar_dataset: Dataset,
                                 output_dir_path: str,
                                 output_format='feather',
                                 size: int=5) -> None:
    """
    This method generates a processed feather dataset from windows feather dataset set dataframe output
    :param oscar_dataset: processed dataset with datafrace output
    :param output_dir_path: path of the feather outputs files
    :param output_format: 'feather' only
    :param size: size of number of positive class to choose. The number of negative class will be the same.
    """

    def  _process_element(idx_ts, ts):
        output_dir = os.path.join(output_dir_path, output_format, 'df_' + str(idx_ts))
        os.makedirs(output_dir, exist_ok=True)
        df_name = 'df_' + str(idx_ts) + '_' + str(idx_ts)
        ts.reset_index(inplace=True)
        ts.to_feather(os.path.join(output_dir, df_name + '.feather'))

    _, events = get_nb_events(oscar_dataset)
    index_positive = [idx for idx, e in enumerate(events) if e == 1]
    index_negative = [idx for idx, e in enumerate(events) if e == 0]
    print(len(index_positive))
    print(len(index_negative))
    positive_choice = random.sample(index_positive, size)
    negative_choice = random.sample(index_negative, size)

    for idx_p in positive_choice:
        _process_element(idx_p, oscar_dataset[idx_p])
    for idx_n in negative_choice:
        _process_element(idx_n, oscar_dataset[idx_n])


def generate_annotations(df: pd.DataFrame, length_event=None, output_events_merge=None):
    """
    Generate annotations from a dataframe containing annotation at one point only.
    :param df: source dataframe used to generate annotation.
    :param length_event: length of the events to complete annotations. format in Offset aliases. \
     None for keeping annotation as-is (default).
    :param output_events_merge: list of ChannelID (not value of ChannelID) to merge to become the ApneaEvent. None for all apnea events
    """
    result = df.copy()
    if output_events_merge:
        possible_apnea_events = output_events_merge
    else:
        possible_apnea_events = [ChannelID.CPAP_ClearAirway, ChannelID.CPAP_Obstructive, ChannelID.CPAP_Hypopnea, ChannelID.CPAP_Apnea]
    possible_apnea_events_str = [c[5] for c in CHANNELS if c[1] in possible_apnea_events]
    events_in_origin = [i for i in result.columns if i in possible_apnea_events_str]
    if len(events_in_origin) == 0:
        result['ApneaEvent'] = 0.0
    else:
        result['ApneaEvent'] = result[events_in_origin].sum(axis=1)
        result['ApneaEvent'] = result['ApneaEvent'].apply(lambda x: 1 if (not pd.isnull(x)) and (x != 0) else np.nan)
        result['ApneaEvent'].fillna(0, inplace=True)
        if length_event is not None:
            list_index_annot = result.index[result['ApneaEvent'] == 1].tolist()
            for annot_index in list_index_annot:
                indexes = result.index[((result.index <= annot_index) &
                                        (result.index >= (annot_index - pd.to_timedelta(length_event))))]
                result.loc[indexes, 'ApneaEvent'] = 1

    return result

def split_dataset(oscar_dataset: Dataset,
                  output_dir_path: str,
                  train_ratio: float,
                  valid_ratio: float):

    len_complete_dataset = len(oscar_dataset)
    # take only a subset of complete dataset
    len_complete_dataset = int(len_complete_dataset * 1.0)
    percentage_split = (train_ratio,
                        valid_ratio,
                        1 - train_ratio - valid_ratio)
    cumul_perc_split = np.cumsum(percentage_split)

    idx_tvt_set = [int(i) for i in (cumul_perc_split * len_complete_dataset)]
    print(idx_tvt_set)

    split_dataset_train = PickleDataset(output_type='numpy',
                                        limits=slice(0, idx_tvt_set[0]),
                                        src_data_path='../data/processing/pickle/')
    split_dataset_valid = PickleDataset(output_type='numpy',
                                        limits=slice(idx_tvt_set[0] + 1, idx_tvt_set[1]),
                                        src_data_path='../data/processing/pickle/'
                                        )
    split_dataset_test = PickleDataset(output_type='numpy',
                                       limits=slice(idx_tvt_set[1] + 1, idx_tvt_set[2]),
                                       src_data_path='../data/processing/pickle/')

    generate_pickle_dataset(split_dataset_train, os.path.join(output_dir_path, 'train'))
    generate_pickle_dataset(split_dataset_valid, os.path.join(output_dir_path, 'valid'))
    generate_pickle_dataset(split_dataset_test, os.path.join(output_dir_path, 'test'))