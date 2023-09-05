from src.data.datasets.processed_dataset import ProcessedDataset
from src.data.utils import get_nb_events

def test___getitem__():
    ds = ProcessedDataset(data_path='../../data/processing/windowed', output_type='dataframe')

    assert len(ds) == 1494
    nb_events, _ = get_nb_events(ds)
    assert nb_events == 1