from datetime import datetime

import numpy as np
import pandas as pd

from src.data.preparation_tasks import align_channels


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
    initial_data = range(0, len(idx)*10, 10)
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
