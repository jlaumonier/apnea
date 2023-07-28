from datetime import datetime

import numpy as np
import pandas as pd

from pyapnea.oscar.oscar_constants import ChannelID

from src.data.preparation_tasks import align_channels, \
    generate_rolling_window_dataframes, generate_annotations


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

    list_result_df = generate_rolling_window_dataframes(original_df, desired_len)

    assert type(list_result_df) == list
    assert len(list_result_df) == 5
    assert len(list_result_df[0]) == desired_len
    assert list_result_df[0].loc['2019-01-01 00:00:00', 'value'] == 9
    assert list_result_df[4].loc['2019-01-01 00:00:08', 'value'] == 73


def test_generate_rolling_window_dataframes_multiple_columns():
    np.random.seed(10)
    desired_len = 2

    rows, cols = 10, 2
    data = np.random.randint(0, 100, size=(rows, cols))
    tidx = pd.date_range('2019-01-01', periods=rows, freq='S')
    original_df = pd.DataFrame(data,
                               columns=['value1', 'value2'], index=tidx)

    list_result_df = generate_rolling_window_dataframes(original_df, desired_len)

    assert type(list_result_df) == list
    assert len(list_result_df) == 5
    assert len(list_result_df[0]) == desired_len
    assert list_result_df[0].loc['2019-01-01 00:00:00', 'value1'] == 9
    assert list_result_df[4].loc['2019-01-01 00:00:08', 'value1'] == 62
    assert list_result_df[0].loc['2019-01-01 00:00:00', 'value2'] == 15
    assert list_result_df[4].loc['2019-01-01 00:00:08', 'value2'] == 33


def test_generate_rolling_window_dataframes_incomplete_df_take_last():
    np.random.seed(10)
    desired_len = 2

    rows, cols = 9, 1
    data = np.random.randint(0, 100, size=(rows, cols))
    tidx = pd.date_range('2019-01-01', periods=rows, freq='S')
    original_df = pd.DataFrame(data,
                               columns=['value'], index=tidx)
    list_result_df = generate_rolling_window_dataframes(original_df, desired_len)

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
    list_result_df = generate_rolling_window_dataframes(original_df, desired_len, keep_last_incomplete=False)

    assert type(list_result_df) == list
    assert len(list_result_df) == 4
    assert len(list_result_df[0]) == desired_len
    assert len(list_result_df[3]) == 2
    assert list_result_df[0].loc['2019-01-01 00:00:00', 'value'] == 9
    assert list_result_df[3].loc['2019-01-01 00:00:07', 'value'] == 8


def test_generate_rolling_window_dataframes_index_not_ordered():
    np.random.seed(10)
    desired_len = 2

    rows, cols = 10, 1
    data = np.random.randint(0, 100, size=(rows, cols))
    tidx = pd.date_range('2019-01-01', periods=rows, freq='S')
    original_df = pd.DataFrame(data,
                               columns=['value'], index=tidx)
    original_df = original_df.sample(frac=1.0)

    list_result_df = generate_rolling_window_dataframes(original_df, desired_len, sort_index=True)

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
    data = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 23, 0]    # Obstructive
    data2 = [0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]    # ClearAirway
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