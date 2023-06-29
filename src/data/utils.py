import pandas as pd


def get_annotations_ends(df: pd.DataFrame):
    """
    get the last timestamps of all annotations in the dataframe
    """
    df_annotation = df[(df['Obstructive'] == 1) &
                       (df['Obstructive'].shift(-1) == 0)]
    if df['Obstructive'].iloc[-1] == 1:
        df_annotation = pd.concat([df_annotation, df.tail(1)])
    return df_annotation
