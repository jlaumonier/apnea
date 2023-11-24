import os
import pytest

from src.data.datasets.slpdb_dataset import SLPDB_Dataset
from src.data.utils import get_nb_events

@pytest.fixture(scope="function")
def relative_path():
    yield '../../'

def test___getitem__(base_directory):
    data_path = os.path.join(base_directory, '..', 'data', 'raw-slpdb/physionet.org/files/slpdb/1.0.0')
    ds = SLPDB_Dataset(data_path=data_path)

    assert len(ds) == 18
    # id elmnt, inputs, first timestep, first sensor
    assert ds[0][0][0][0] == -60.0
    # # id elmnt, class
    assert ds[0][1][0] == 0
    #
    nb_events, events = get_nb_events(ds)
    assert nb_events == 13