import os
import pytest

from src.data.datasets.raw_oscar_dataset import RawOscarDataset
from src.data.utils import get_nb_events

@pytest.fixture(scope="function")
def relative_path():
    yield '../../'

def test___getitem__(base_directory):
    data_path = os.path.join(base_directory, 'data', 'raw')
    ds = RawOscarDataset(data_path=data_path)

    assert len(ds) == 2
    # id elmnt, inputs, first timestep, first sensor
    assert ds[0][0][0][0] == -38.76000154018402
    # id elmnt, class
    assert ds[0][1][0] == 0

    nb_events, events = get_nb_events(ds)
    assert nb_events == 1