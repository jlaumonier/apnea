from src.data.datasets.slpdb_dataset import SLPDB_Dataset
from src.data.utils import get_nb_events

def test___getitem__():

    ds = SLPDB_Dataset(data_path='../../../data/raw-slpdb/physionet.org/files/slpdb/1.0.0')

    assert len(ds) == 18
    # id elmnt, inputs, first timestep, first sensor
    assert ds[0][0][0][0] == -60.0
    # # id elmnt, class
    assert ds[0][1][0] == 0
    #
    nb_events, events = get_nb_events(ds)
    assert nb_events == 13