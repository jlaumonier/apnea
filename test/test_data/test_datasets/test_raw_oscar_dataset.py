from src.data.datasets.raw_oscar_dataset import RawOscarDataset
from src.data.utils import get_nb_events

def test___getitem__():

    ds = RawOscarDataset(data_path='../../data/raw')

    assert len(ds) == 2
    # id elmnt, inputs, first timestep, first sensor
    assert ds[0][0][0][0] == -38.76000154018402
    # id elmnt, class
    assert ds[0][1][0] == 0

    nb_events, events = get_nb_events(ds)
    assert nb_events == 1