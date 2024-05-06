import os

import pytest
from src.data.datasets.processed_dataset import ProcessedDataset
from src.data.utils import get_nb_events

@pytest.fixture(scope="function")
def relative_path():
    yield '../../'

def test___getitem___feather_dataframe(base_directory):
    data_path = os.path.join(base_directory, 'data', 'repository', 'datasets', '9e81da40-41a1-4f9b-9bba-41de71b0ebd9')
    ds = ProcessedDataset(data_path=data_path, getitem_type='dataframe')

    assert len(ds) == 1494
    nb_events, _ = get_nb_events(ds)
    assert nb_events == 24

def test___getitem___pickle_numpy(base_directory):
    data_path = os.path.join(base_directory, 'data', 'repository', 'datasets', 'a67ff056-fd10-4b41-bd1a-104f9e23279e')
    ds = ProcessedDataset(data_path=data_path, getitem_type='numpy')

    assert len(ds) == 1494
    nb_events, _ = get_nb_events(ds)
    assert nb_events == 24

def test__getitem___compressed_feather_dataframe(base_directory):
    data_path = os.path.join(base_directory, 'data', 'repository', 'datasets', '9e81da40-41a1-4f9b-9bba-41de71b0ebd9')
    ds = ProcessedDataset(data_path=data_path, getitem_type='dataframe')

    assert len(ds) == 1494
    nb_events, _ = get_nb_events(ds)
    assert nb_events == 24
