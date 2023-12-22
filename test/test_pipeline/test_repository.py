import os
import shutil
import pytest

from omegaconf import OmegaConf

from src.pipeline.repository import Repository
from src.data.datasets.raw_oscar_dataset import RawOscarDataset

@pytest.fixture(scope="function")
def relative_path():
    yield '../'


def test_repository_init_create_repo(base_directory):
    os.makedirs(os.path.join(base_directory, 'data', 'temp'), exist_ok=True)

    data_repo_path = os.path.join(base_directory, 'data', 'temp', 'repo')

    assert not os.path.exists(data_repo_path)

    repo = Repository(data_repo_path)

    assert repo.path == data_repo_path
    assert os.path.isdir(data_repo_path)
    assert os.path.isfile(os.path.join(data_repo_path, 'metadata_db.json'))
    assert repo.valid_repo == 0

    shutil.rmtree(os.path.join(base_directory, 'data', 'temp'))

def test_repository_init_repo_loaded(base_directory):
    data_repo_path = os.path.join(base_directory, 'data', 'repository')

    repo = Repository(data_repo_path)

    assert repo.valid_repo == 0
    assert '8b663706-ab51-4a9a-9a66-eb9ac2c135f3' in repo.metadata['datasets'].keys()

def test_repository_boostrap_copy(base_directory):
    os.makedirs(os.path.join(base_directory, 'data', 'temp'), exist_ok=True)

    data_repo_path = os.path.join(base_directory, 'data', 'temp', 'repository')
    source_dataset_path = os.path.join(base_directory, 'data', 'raw')

    repo = Repository(data_repo_path)
    guid = repo.bootstrap(source_dataset_path, RawOscarDataset)

    conf = OmegaConf.load(os.path.join(data_repo_path, 'conf', str(guid)+'.yaml'))
    expected_type = 'src.data.datasets.raw_oscar_dataset.RawOscarDataset'

    assert str(guid) in repo.metadata['datasets'].keys()
    assert repo.metadata['datasets'][str(guid)]['type'] == expected_type
    assert os.path.exists(os.path.join(data_repo_path, 'datasets', str(guid)))
    assert os.path.isfile(os.path.join(data_repo_path, 'conf', str(guid)+'.yaml'))
    assert conf['_target_'] == expected_type
    assert repo.valid_repo == 0

    shutil.rmtree(os.path.join(base_directory, 'data', 'temp'))

def test_load_dataset(base_directory):
    data_repo_path = os.path.join(base_directory, 'data', 'repository')

    repo = Repository(data_repo_path)
    dataset = repo.load_dataset('8b663706-ab51-4a9a-9a66-eb9ac2c135f3', 'dataframe')

    assert dataset.__class__.__name__ ==  'RawOscarDataset'
    assert len(dataset) == 2

def test_create_dataset(base_directory):
    os.makedirs(os.path.join(base_directory, 'data', 'temp'), exist_ok=True)
    data_repo_path = os.path.join(base_directory, 'data', 'temp', 'repository')
    repo = Repository(data_repo_path)

    guid, dest_data_path = repo.create_dataset()

    assert str(guid) in dest_data_path

    shutil.rmtree(os.path.join(base_directory, 'data', 'temp'))

def test_commit_repository(base_directory):
    os.makedirs(os.path.join(base_directory, 'data', 'temp'), exist_ok=True)
    data_repo_path = os.path.join(base_directory, 'data', 'temp', 'repository')
    repo = Repository(data_repo_path)
    guid, dest_data_path = repo.create_dataset()
    task_config = OmegaConf.create()
    task_config['test'] = 'test1'

    repo.commit_dataset(guid, RawOscarDataset, task_config)

    assert os.path.exists(os.path.join(data_repo_path, 'conf', str(guid)+'.yaml'))
    tested_conf = OmegaConf.load(os.path.join(data_repo_path, 'conf', str(guid)+'.yaml'))
    assert 'RawOscarDataset' in tested_conf['_target_']
    assert tested_conf['data_path'] == '??'
    assert str(guid) in repo.metadata['datasets']
    assert 'RawOscarDataset' in repo.metadata['datasets'][str(guid)]['type']
    assert repo.valid_repo

    shutil.rmtree(os.path.join(base_directory, 'data', 'temp'))