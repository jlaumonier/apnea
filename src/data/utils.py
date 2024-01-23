from tqdm import tqdm
import pandas as pd


def get_annotations_ends(df: pd.DataFrame):
    """
    get the last timestamps of all annotations in the dataframe
    """
    df_annotation = df[(df['ApneaEvent'] == 1) &
                       (df['ApneaEvent'].shift(-1) == 0)]
    if df['ApneaEvent'].iloc[-1] == 1:
        df_annotation = pd.concat([df_annotation, df.tail(1)])
    return df_annotation

def is_contain_event(element, output_type='dataframe'):
    """
    return True if the element of a dataset contains at least one event
    :para element: element to consider
    :param output_type: 'dataframe' : an element is a dataframe containing at least the 'ApneaEvent' column
                        'numpy' : an element is a tuple (x, y)
    """
    if output_type == 'dataframe':
        return (1.0 in element['ApneaEvent'].values)
    else:
        return (1.0 in element[1])


def get_nb_events(dataset):
    """
    Get the number of element of a dataset that contain at least one apnea event
    de plus : https://stackoverflow.com/questions/50756085/how-to-print-the-progress-of-a-list-comprehension-in-python
    """

    events = [is_contain_event(element, dataset.getitem_type) for element in tqdm(dataset)]
    nb_contain_event = events.count(1)
    return nb_contain_event, events
