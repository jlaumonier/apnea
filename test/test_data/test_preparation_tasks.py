import os
import shutil
import pytest
from datetime import datetime

import numpy as np
import pandas as pd

from pyapnea.oscar.oscar_constants import ChannelID
from pyapnea.pytorch.raw_oscar_dataset import RawOscarDataset

from src.data.datasets.processed_dataset import ProcessedDataset
from src.data.preparation_tasks import align_channels, \
    generate_rolling_window_dataframes, generate_annotations, task_generate_all_rolling_window, \
    task_generate_balanced_dataset, task_generate_pickle_dataset, task_generate_split_dataset


@pytest.fixture(scope="function")
def relative_path():
    yield '../'


def test_align_channels_non_aligned():
    idx = pd.date_range(start='2023-06-15 00:00:00', end='2023-06-15 00:00:00.2', freq='100ms')
    initial_data = range(len(idx))
    df = pd.DataFrame(data=list(zip(idx, initial_data)), columns=['time_utc', 'col1'])

    new_row = [datetime(2023, 6, 15, 00, 00, 00, 70000), 1]
    new_df = pd.DataFrame(data=[new_row], columns=['time_utc', 'col2'])

    df = pd.merge(df, new_df, on='time_utc', how='outer')
    df = df.sort_values(by='time_utc')

    result_df = align_channels(df, reference_channel='col1', period_ref_channel='40ms')

    assert result_df.loc[result_df['col2'] == 1]['time_utc'].item() == pd.Timestamp('2023-06-15 00:00:00.100000')


def test_align_channels_non_aligned_onw_raw_aligned():
    idx = pd.date_range(start='2023-06-15 00:00:00', end='2023-06-15 00:00:00.3', freq='100ms')
    initial_data = range(len(idx))
    df = pd.DataFrame(data=list(zip(idx, initial_data)), columns=['time_utc', 'col1'])

    new_rows = [[datetime(2023, 6, 15, 00, 00, 00, 70000), 1],
                [datetime(2023, 6, 15, 00, 00, 00, 200000), 1]]
    new_df = pd.DataFrame(data=new_rows, columns=['time_utc', 'col2'])
    df = pd.merge(df, new_df, on='time_utc', how='outer')
    df = df.sort_values(by='time_utc')

    result_df = align_channels(df, reference_channel='col1', period_ref_channel='40ms')

    assert result_df.loc[result_df['col2'] == 1]['time_utc'].to_list() == [pd.Timestamp('2023-06-15 00:00:00.100000'),
                                                                           pd.Timestamp('2023-06-15 00:00:00.200000')]


def test_align_channels_multiple_channels():
    idx = pd.date_range(start='2023-06-15 00:00:00', end='2023-06-15 00:00:00.3', freq='100ms')
    initial_data = range(0, len(idx) * 10, 10)
    df = pd.DataFrame(data=list(zip(idx, initial_data)), columns=['time_utc', 'col1'])

    new_rows = [[datetime(2023, 6, 15, 00, 00, 00, 70000), 1, 2],
                [datetime(2023, 6, 15, 00, 00, 00, 200000), 1, np.nan],
                [datetime(2023, 6, 15, 00, 00, 00, 210000), np.nan, 2],
                [datetime(2023, 6, 15, 00, 00, 00, 310000), np.nan, 2]]

    new_df = pd.DataFrame(data=new_rows, columns=['time_utc', 'col2', 'col3'])
    df = pd.merge(df, new_df, on='time_utc', how='outer')
    df = df.sort_values(by='time_utc')

    result_df = align_channels(df, reference_channel='col1', period_ref_channel='40ms')

    assert result_df.loc[(result_df['col2'] == 1) |
                         (result_df['col3'] == 2)]['time_utc'].to_list() == [pd.Timestamp('2023-06-15 00:00:00.100000'),
                                                                             pd.Timestamp('2023-06-15 00:00:00.200000'),
                                                                             pd.Timestamp('2023-06-15 00:00:00.300000')]


def test_generate_rolling_window_dataframes_perfect_df():
    np.random.seed(10)
    desired_len = 2

    rows, cols = 10, 1
    data = np.random.randint(0, 100, size=(rows, cols))
    tidx = pd.date_range('2019-01-01', periods=rows, freq='S')
    original_df = pd.DataFrame(data,
                               columns=['value'], index=tidx)

    list_result_df = generate_rolling_window_dataframes(df=original_df,
                                                        length=desired_len,
                                                        step=desired_len)

    assert type(list_result_df) == list
    assert len(list_result_df) == 5
    assert len(list_result_df[0]) == desired_len
    assert list_result_df[0].loc['2019-01-01 00:00:00', 'value'] == 9
    assert list_result_df[4].loc['2019-01-01 00:00:08', 'value'] == 73


def test_generate_rolling_window_dataframes_perfect_df_with_step():
    np.random.seed(10)
    desired_len = 4
    step = 2

    rows, cols = 10, 1
    data = np.random.randint(0, 100, size=(rows, cols))
    tidx = pd.date_range('2019-01-01', periods=rows, freq='S')
    original_df = pd.DataFrame(data,
                               columns=['value'], index=tidx)

    list_result_df = generate_rolling_window_dataframes(df=original_df,
                                                        length=desired_len,
                                                        step=step)
    assert type(list_result_df) == list
    assert len(list_result_df) == 4
    assert len(list_result_df[0]) == desired_len
    assert list_result_df[0].loc['2019-01-01 00:00:00', 'value'] == 9
    assert list_result_df[3].loc['2019-01-01 00:00:08', 'value'] == 73


def test_generate_rolling_window_dataframes_multiple_columns():
    np.random.seed(10)
    desired_len = 2

    rows, cols = 10, 2
    data = np.random.randint(0, 100, size=(rows, cols))
    tidx = pd.date_range('2019-01-01', periods=rows, freq='S')
    original_df = pd.DataFrame(data,
                               columns=['value1', 'value2'], index=tidx)

    list_result_df = generate_rolling_window_dataframes(df=original_df,
                                                        length=desired_len,
                                                        step=desired_len)

    assert type(list_result_df) == list
    assert len(list_result_df) == 5
    assert len(list_result_df[0]) == desired_len
    assert list_result_df[0].loc['2019-01-01 00:00:00', 'value1'] == 9
    assert list_result_df[4].loc['2019-01-01 00:00:08', 'value1'] == 62
    assert list_result_df[0].loc['2019-01-01 00:00:00', 'value2'] == 15
    assert list_result_df[4].loc['2019-01-01 00:00:08', 'value2'] == 33

def test_generate_rolling_window_dataframes_multiple_columns_one_point():
    np.random.seed(10)
    desired_len = 3
    step = 2

    rows, cols = 10, 2
    data = np.random.randint(0, 100, size=(rows, cols))
    tidx = pd.date_range('2019-01-01', periods=rows, freq='S')
    original_df = pd.DataFrame(data,
                               columns=['value1', 'ApneaEvent'], index=tidx)
    original_df['ApneaEvent'] = np.where(original_df['ApneaEvent'] > 50, 1, 0)

    list_result_df = generate_rolling_window_dataframes(df=original_df,
                                                        length=desired_len,
                                                        step=step,
                                                        annotation_type='ONE_POINT',
                                                        one_point_annot_duration=2)
    assert type(list_result_df) == list
    assert type(list_result_df[0]) == tuple
    assert len(list_result_df) == 5
    assert len(list_result_df[0][0]) == desired_len
    assert list_result_df[0][0].loc['2019-01-01 00:00:00', 'value1'] == 9
    assert list_result_df[4][0].loc['2019-01-01 00:00:08', 'value1'] == 62
    assert list_result_df[0][1] == 0
    assert list_result_df[2][1] == 1
    assert list_result_df[3][1] == 1
    assert list_result_df[4][1] == 0


def test_generate_rolling_window_dataframes_incomplete_df_take_last():
    np.random.seed(10)
    desired_len = 2

    rows, cols = 9, 1
    data = np.random.randint(0, 100, size=(rows, cols))
    tidx = pd.date_range('2019-01-01', periods=rows, freq='S')
    original_df = pd.DataFrame(data,
                               columns=['value'], index=tidx)

    list_result_df = generate_rolling_window_dataframes(df=original_df,
                                                        length=desired_len,
                                                        step=desired_len)
    assert type(list_result_df) == list
    assert len(list_result_df) == 5
    assert len(list_result_df[0]) == desired_len
    assert len(list_result_df[4]) == 1
    assert list_result_df[0].loc['2019-01-01 00:00:00', 'value'] == 9
    assert list_result_df[4].loc['2019-01-01 00:00:08', 'value'] == 73


def test_generate_rolling_window_dataframes_incomplete_df_not_take_last():
    np.random.seed(10)
    desired_len = 2

    rows, cols = 9, 1
    data = np.random.randint(0, 100, size=(rows, cols))
    tidx = pd.date_range('2019-01-01', periods=rows, freq='S')
    original_df = pd.DataFrame(data,
                               columns=['value'], index=tidx)

    list_result_df = generate_rolling_window_dataframes(df=original_df,
                                                        length=desired_len,
                                                        keep_last_incomplete=False,
                                                        step=desired_len)

    assert type(list_result_df) == list
    assert len(list_result_df) == 4
    assert len(list_result_df[0]) == desired_len
    assert len(list_result_df[3]) == 2
    assert list_result_df[0].loc['2019-01-01 00:00:00', 'value'] == 9
    assert list_result_df[3].loc['2019-01-01 00:00:07', 'value'] == 8


def test_generate_rolling_window_dataframes_incomplete_df_take_last_with_step():
    np.random.seed(10)
    desired_len = 3
    step = 2

    rows, cols = 10, 1
    data = np.random.randint(0, 100, size=(rows, cols))
    tidx = pd.date_range('2019-01-01', periods=rows, freq='S')
    original_df = pd.DataFrame(data,
                               columns=['value'], index=tidx)

    list_result_df = generate_rolling_window_dataframes(df=original_df,
                                                        length=desired_len,
                                                        keep_last_incomplete=True,
                                                        step=step)

    assert type(list_result_df) == list
    assert len(list_result_df) == 5
    assert len(list_result_df[0]) == desired_len
    assert len(list_result_df[4]) == 2
    assert list_result_df[0].loc['2019-01-01 00:00:00', 'value'] == 9
    assert list_result_df[4].loc['2019-01-01 00:00:09', 'value'] == 0


def test_generate_rolling_window_dataframes_no_incomplete_df_take_last_with_step():
    np.random.seed(10)
    desired_len = 3
    step = 2

    rows, cols = 9, 1
    data = np.random.randint(0, 100, size=(rows, cols))
    tidx = pd.date_range('2019-01-01', periods=rows, freq='S')
    original_df = pd.DataFrame(data,
                               columns=['value'], index=tidx)

    list_result_df = generate_rolling_window_dataframes(df=original_df,
                                                        length=desired_len,
                                                        keep_last_incomplete=True,
                                                        step=step)

    assert type(list_result_df) == list
    assert len(list_result_df) == 4
    assert len(list_result_df[0]) == desired_len
    assert len(list_result_df[3]) == 3
    assert list_result_df[0].loc['2019-01-01 00:00:00', 'value'] == 9
    assert list_result_df[3].loc['2019-01-01 00:00:08', 'value'] == 73


def test_generate_rolling_window_dataframes_index_not_ordered():
    np.random.seed(10)
    desired_len = 2

    rows, cols = 10, 1
    data = np.random.randint(0, 100, size=(rows, cols))
    tidx = pd.date_range('2019-01-01', periods=rows, freq='S')
    original_df = pd.DataFrame(data,
                               columns=['value'], index=tidx)
    original_df = original_df.sample(frac=1.0)

    list_result_df = generate_rolling_window_dataframes(df=original_df,
                                                        length=desired_len,
                                                        step=desired_len,
                                                        sort_index=True)

    assert type(list_result_df) == list
    assert len(list_result_df) == 5
    assert len(list_result_df[0]) == desired_len
    assert list_result_df[0].loc['2019-01-01 00:00:00', 'value'] == 9
    assert list_result_df[3].loc['2019-01-01 00:00:07', 'value'] == 8


def test_generate_annotation_keep():
    np.random.seed(10)

    rows, cols = 9, 1
    data = np.random.randint(0, 2, size=(rows, cols))
    tidx = pd.date_range('2019-01-01', periods=rows, freq='S')
    original_df = pd.DataFrame(data,
                               columns=['Obstructive'], index=tidx)

    result_df = generate_annotations(original_df)

    assert (result_df['Obstructive'].isin([0, 1]).sum(axis=0) == len(result_df))
    assert (result_df['Obstructive'].to_list() == original_df['Obstructive'].to_list())
    assert (result_df['ApneaEvent'].to_list() == result_df['Obstructive'].to_list())


def test_generate_annotation_1_nan():
    np.random.seed(10)

    rows, cols = 6, 1
    data = [np.nan, 10, np.nan, np.nan, 0.5, np.nan]
    tidx = pd.date_range('2019-01-01', periods=rows, freq='S')
    original_df = pd.DataFrame(data,
                               columns=['Obstructive'], index=tidx)

    result_df = generate_annotations(original_df)

    assert result_df['Obstructive'].equals(original_df['Obstructive'])
    assert result_df['ApneaEvent'].to_list() == [0, 1, 0, 0, 1, 0]


def test_generate_annotation_no_event():
    np.random.seed(10)

    rows, cols = 6, 1
    data = np.random.randint(0, 2, size=(rows, cols))
    tidx = pd.date_range('2019-01-01', periods=rows, freq='S')
    original_df = pd.DataFrame(data,
                               columns=['FlowRate'], index=tidx)

    result_df = generate_annotations(original_df)

    assert (result_df['ApneaEvent'].isin([0, 1]).sum(axis=0) == len(result_df))
    assert (result_df['ApneaEvent'].to_list() == ([0.0] * rows))
    assert result_df.columns.tolist() == ['FlowRate', 'ApneaEvent']


def test_generate_annotation_entire_event():
    np.random.seed(10)

    rows, cols = 20, 1
    data = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 23, 0]
    tidx = pd.date_range('2019-01-01', periods=rows, freq='S')
    original_df = pd.DataFrame(data,
                               columns=['Obstructive'], index=tidx)

    result_df = generate_annotations(original_df, length_event='10S')

    assert (result_df['ApneaEvent'].isin([0, 1]).sum(axis=0) == len(result_df))
    assert (result_df['ApneaEvent'].to_list() == [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0])
    assert (result_df['Obstructive'].to_list() == original_df['Obstructive'].to_list())


def test_generate_annotation_entire_event_size_lower():
    np.random.seed(10)

    rows, cols = 5, 1
    data = [0, 0, 0, 23, 0]
    tidx = pd.date_range('2019-01-01', periods=rows, freq='S')
    original_df = pd.DataFrame(data,
                               columns=['Obstructive'], index=tidx)

    result_df = generate_annotations(original_df, length_event='10S')

    assert (result_df['ApneaEvent'].isin([0, 1]).sum(axis=0) == len(result_df))
    assert (result_df['ApneaEvent'].to_list() == [1, 1, 1, 1, 0])
    assert (result_df['Obstructive'].to_list() == original_df['Obstructive'].to_list())


def test_generate_annotation_multi_event():
    np.random.seed(10)

    rows, cols = 20, 2
    data = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 23, 0]
    data2 = [0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tidx = pd.date_range('2019-01-01', periods=rows, freq='S')
    original_df = pd.DataFrame(data,
                               columns=['Obstructive'], index=tidx)
    original_df['ClearAirway'] = data2

    result_df = generate_annotations(original_df, length_event='10S')

    assert (result_df['ApneaEvent'].isin([0, 1]).sum(axis=0) == len(result_df))
    assert (result_df['ApneaEvent'].to_list() == [1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0])
    assert (result_df['Obstructive'].to_list() == original_df['Obstructive'].to_list())
    assert (result_df['ClearAirway'].to_list() == original_df['ClearAirway'].to_list())


def test_generate_annotation_multi_event_merge_some():
    np.random.seed(10)

    rows, cols = 20, 2
    data = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 23, 0]  # Obstructive
    data2 = [0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # ClearAirway
    data3 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # Hypopnea
    tidx = pd.date_range('2019-01-01', periods=rows, freq='S')
    original_df = pd.DataFrame(data,
                               columns=['Obstructive'], index=tidx)
    original_df['ClearAirway'] = data2
    original_df['Hypopnea'] = data3

    result_df = generate_annotations(original_df,
                                     length_event='4S',
                                     output_events_merge=[ChannelID.CPAP_ClearAirway,
                                                          ChannelID.CPAP_Hypopnea])

    assert (result_df['ApneaEvent'].isin([0, 1]).sum(axis=0) == len(result_df))
    assert (result_df['ApneaEvent'].to_list() == [1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    assert (result_df['Obstructive'].to_list() == original_df['Obstructive'].to_list())
    assert (result_df['ClearAirway'].to_list() == original_df['ClearAirway'].to_list())
    assert (result_df['Hypopnea'].to_list() == original_df['Hypopnea'].to_list())


# -- TASKS --

def test_task_generate_all_rolling_window_output_feather_df(base_directory):
    data_path = os.path.join(base_directory, 'data')
    os.makedirs(os.path.join(data_path, 'temp'), exist_ok=True)
    oscar_dataset = RawOscarDataset(data_path=os.path.join(data_path, 'raw'),
                                    getitem_type='dataframe', limits=None)

    type_ds, file_format = task_generate_all_rolling_window(oscar_dataset=oscar_dataset,
                                                            length=500,
                                                            keep_last_incomplete=False,
                                                            output_dir_path=os.path.join(data_path, 'temp',
                                                                                         'processing',
                                                                                         'windowed'),
                                                            step=500)

    assert type_ds == ProcessedDataset
    assert file_format == 'feather'

    assert len(os.listdir(os.path.join(data_path, 'temp', 'processing', 'windowed', 'feather'))) == 2
    assert len(os.listdir(os.path.join(data_path, 'temp', 'processing', 'windowed', 'feather', 'df_0'))) == 1443
    assert len(os.listdir(os.path.join(data_path, 'temp', 'processing', 'windowed', 'feather', 'df_1'))) == 51

    dfs_path = os.path.join(data_path, 'temp', 'processing', 'windowed', 'feather', 'df_0')
    for filename in os.listdir(dfs_path):
        df = pd.read_feather(os.path.join(dfs_path, filename))
        assert 'time_utc' in df.keys()
        assert 'FlowRate' in df.keys()
        assert 'ApneaEvent' in df.keys()
        assert df['time_utc'].is_monotonic_increasing

    shutil.rmtree(os.path.join(data_path, 'temp'))

def test_task_generate_all_rolling_window_one_point(base_directory):
    data_path = os.path.join(base_directory, 'data')
    os.makedirs(os.path.join(data_path, 'temp'), exist_ok=True)
    oscar_dataset = RawOscarDataset(data_path=os.path.join(data_path, 'raw'),
                                    getitem_type='dataframe', limits=None)

    type_ds, file_format = task_generate_all_rolling_window(oscar_dataset=oscar_dataset,
                                                            length=250,
                                                            keep_last_incomplete=False,
                                                            output_dir_path=os.path.join(data_path, 'temp',
                                                                                         'processing',
                                                                                         'windowed'),
                                                            step=150,
                                                            annotation_type='ONE_POINT',
                                                            one_point_annot_duration=80
                                                            )

    assert type_ds == ProcessedDataset

    assert sorted(os.listdir(os.path.join(data_path, 'temp', 'processing', 'windowed'))) == ['data.feather',
                                                                                             'feather']
    assert len(os.listdir(os.path.join(data_path, 'temp', 'processing', 'windowed', 'feather'))) == 2
    assert len(os.listdir(os.path.join(data_path, 'temp', 'processing', 'windowed', 'feather', 'df_0'))) == 4809
    assert len(os.listdir(os.path.join(data_path, 'temp', 'processing', 'windowed', 'feather', 'df_1'))) == 169

    df = pd.read_feather(os.path.join(data_path, 'temp', 'processing', 'windowed', 'data.feather'))
    assert len(df) == 4809 + 169
    assert df.iloc[0].to_list() == ['feather/df_0/df_0_0.feather', 0]

    shutil.rmtree(os.path.join(data_path, 'temp'))


def test_task_generate_balanced_dataset_dataframe_feather(base_directory):
    data_path = os.path.join(base_directory, 'data')
    os.makedirs(os.path.join(data_path, 'temp'), exist_ok=True)

    processed_dataset = ProcessedDataset(
        data_path=os.path.join(data_path, 'repository', 'datasets', '9e81da40-41a1-4f9b-9bba-41de71b0ebd9'),
        getitem_type='dataframe', limits=None)

    type_ds, file_format = task_generate_balanced_dataset(oscar_dataset=processed_dataset,
                                                          output_format='same',
                                                          size=1,
                                                          output_dir_path=os.path.join(data_path, 'temp', 'processing',
                                                                                       'balanced'))

    assert type_ds == ProcessedDataset
    assert file_format == processed_dataset.file_format

    list_dir = os.listdir(os.path.join(data_path, 'temp', 'processing', 'balanced', 'feather'))
    assert len(list_dir) == 2
    assert len(os.listdir(os.path.join(data_path, 'temp', 'processing', 'balanced', 'feather', list_dir[0]))) == 1

    shutil.rmtree(os.path.join(data_path, 'temp'))


def test_task_generate_balanced_dataset_numpy_pickle(base_directory):
    data_path = os.path.join(base_directory, 'data')
    os.makedirs(os.path.join(data_path, 'temp'), exist_ok=True)

    processed_dataset = ProcessedDataset(
        data_path=os.path.join(data_path, 'repository', 'datasets', 'a67ff056-fd10-4b41-bd1a-104f9e23279e'),
        getitem_type='numpy', limits=None)

    type_ds, file_format = task_generate_balanced_dataset(oscar_dataset=processed_dataset,
                                                          output_format='same',
                                                          size=1,
                                                          output_dir_path=os.path.join(data_path, 'temp', 'processing',
                                                                                       'balanced'))

    assert type_ds == ProcessedDataset
    assert file_format == processed_dataset.file_format

    list_files = os.listdir(os.path.join(data_path, 'temp', 'processing', 'balanced'))
    list_files.sort()
    assert len(list_files) == 2
    assert 'gt.pkl' in os.path.join(data_path, 'temp', 'processing', 'balanced', list_files[0])
    assert 'inputs.pkl' in os.path.join(data_path, 'temp', 'processing', 'balanced', list_files[1])

    shutil.rmtree(os.path.join(data_path, 'temp'))

def test_task_generate_pickle(base_directory):
    data_path = os.path.join(base_directory, 'data')
    os.makedirs(os.path.join(data_path, 'temp'), exist_ok=True)
    processed_dataset = ProcessedDataset(
        data_path=os.path.join(data_path, 'repository', 'datasets', '9e81da40-41a1-4f9b-9bba-41de71b0ebd9'),
        getitem_type='numpy', limits=None)

    type_ds, file_format = task_generate_pickle_dataset(processed_dataset,
                                                        output_dir_path=os.path.join(data_path, 'temp', 'processing',
                                                                                     'pickle'))

    assert len(os.listdir(os.path.join(data_path, 'temp', 'processing', 'pickle'))) == 2
    assert type_ds == ProcessedDataset
    assert file_format == 'pickle'

    shutil.rmtree(os.path.join(data_path, 'temp'))


def test_task_generate_split_dataset(base_directory):
    data_path = os.path.join(base_directory, 'data')
    os.makedirs(os.path.join(data_path, 'temp'), exist_ok=True)
    processed_dataset = ProcessedDataset(
        data_path=os.path.join(data_path, 'repository', 'datasets', 'a67ff056-fd10-4b41-bd1a-104f9e23279e'),
        getitem_type='numpy', limits=None)

    type_ds, file_format = task_generate_split_dataset(processed_dataset,
                                                       output_dir_path=os.path.join(data_path, 'temp', 'processing',
                                                                                    'split'),
                                                       train_ratio=0.8,
                                                       valid_ratio=0.1)

    assert len(os.listdir(os.path.join(data_path, 'temp', 'processing', 'split'))) == 3
    assert type_ds == ProcessedDataset
    assert file_format == 'pickle'

    shutil.rmtree(os.path.join(data_path, 'temp'))
