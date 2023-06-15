import pandas as pd


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
        result_df[c] = result_df[c+'_x'].combine_first(result_df[c+'_y'])
        result_df.drop([c+'_x', c+'_y'], axis=1, inplace=True)
    return result_df