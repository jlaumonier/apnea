import os

import pandas as pd
import numpy as np
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
    data_path = os.path.join(base_directory, 'data', 'repository', 'datasets', '46bdf022-fa33-471d-8e11-3cc6cb60574f')
    ds = ProcessedDataset(data_path=data_path, getitem_type='dataframe')

    assert len(ds) == 1494
    nb_events, _ = get_nb_events(ds)
    assert nb_events == 24

def test___getitem___feather_dataframe_one_point(base_directory):
    data_path = os.path.join(base_directory, 'data', 'repository', 'datasets', '8a406376-2bf3-4430-9d61-54d8eb6099b2')
    ds = ProcessedDataset(data_path=data_path, getitem_type='dataframe')

    assert len(ds) == 4978
    assert type(ds[0]) == tuple
    assert type(ds[0][0]) == pd.DataFrame
    assert type(ds[0][1]) == np.float32
    nb_events = len([e[1] for e in ds if e[1]==1])
    assert nb_events == 36


def test___getitem___feather_numpy_one_point(base_directory):
    data_path = os.path.join(base_directory, 'data', 'repository', 'datasets', '8a406376-2bf3-4430-9d61-54d8eb6099b2')
    ds = ProcessedDataset(data_path=data_path, getitem_type='numpy')

    assert len(ds) == 4978
    assert type(ds[0]) == tuple
    assert type(ds[0][0]) == np.ndarray
    assert type(ds[0][1]) == np.float32
    nb_events = len([e[1] for e in ds if e[1] == 1])
    assert nb_events == 36

def test___getitem___pickle_numpy_one_point(base_directory):
    data_path = os.path.join(base_directory, 'data', 'repository', 'datasets', '5b94033a-581f-48de-afae-677fa00f772a')
    ds = ProcessedDataset(data_path=data_path, getitem_type='numpy')

    assert len(ds) == 4978
    assert type(ds[0]) == tuple
    assert type(ds[0][0]) == np.ndarray
    assert type(ds[0][1]) == np.float32
    nb_events = len([e[1] for e in ds if e[1] == 1])
    assert nb_events == 36
