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
    file_format = 'raw'

    repo = Repository(data_repo_path)
    guid = repo.bootstrap(source_dataset_path, RawOscarDataset, file_format)

    conf = OmegaConf.load(os.path.join(data_repo_path, 'conf', str(guid)+'.yaml'))
    expected_type = 'src.data.datasets.raw_oscar_dataset.RawOscarDataset'

    assert str(guid) in repo.metadata['datasets'].keys()
    assert repo.metadata['datasets'][str(guid)]['type'] == expected_type
    assert repo.metadata['datasets'][str(guid)]['file_format'] == file_format
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

def test_load_dataset_sub_dataset(base_directory):
    data_repo_path = os.path.join(base_directory, 'data', 'repository')

    repo = Repository(data_repo_path)
    dataset = repo.load_dataset('eb0e1bf4-88cf-4cfb-850c-ad11092af8f7', 'dataframe', sub_dataset='train')

    assert dataset.__class__.__name__ == 'PickleDataset'
    assert len(dataset) == 1195

def test_create_dataset(base_directory):
    os.makedirs(os.path.join(base_directory, 'data', 'temp'), exist_ok=True)
    data_repo_path = os.path.join(base_directory, 'data', 'temp', 'repository')
    repo = Repository(data_repo_path)

    guid, dest_data_path = repo.create_dataset()

    assert str(guid) in dest_data_path

    shutil.rmtree(os.path.join(base_directory, 'data', 'temp'))

def test_commit_dataset(base_directory):
    os.makedirs(os.path.join(base_directory, 'data', 'temp'), exist_ok=True)
    data_repo_path = os.path.join(base_directory, 'data', 'temp', 'repository')
    repo = Repository(data_repo_path)
    guid, dest_data_path = repo.create_dataset()
    task_config = OmegaConf.create()
    task_config['test'] = 'test1'
    file_format = 'raw'

    # create a fake dataset content
    os.makedirs(dest_data_path)
    fp = open(os.path.join(dest_data_path,'temp.txt'), 'w')
    fp.write('first line')
    fp.close()

    repo.commit_dataset(guid, RawOscarDataset, file_format, task_config)

    assert os.path.exists(os.path.join(data_repo_path, 'conf', str(guid)+'.yaml'))
    tested_conf = OmegaConf.load(os.path.join(data_repo_path, 'conf', str(guid)+'.yaml'))
    assert 'RawOscarDataset' in tested_conf['_target_']
    assert tested_conf['data_path'] == '??'
    assert str(guid) in repo.metadata['datasets']
    assert 'RawOscarDataset' in repo.metadata['datasets'][str(guid)]['type']
    assert repo.metadata['datasets'][str(guid)]['file_format'] == file_format
    repo._valid_repository()
    assert repo.valid_repo == 0
    # ensure that metadata.json has been saved
    repo._load_repository()
    assert str(guid) in repo.metadata['datasets']

    shutil.rmtree(os.path.join(base_directory, 'data', 'temp'))

def test_remove_dataset_leaf(base_directory):
    src_data_repo_path = os.path.join(base_directory, 'data', 'repository')
    os.makedirs(os.path.join(base_directory, 'data', 'temp'), exist_ok=True)
    temp_data_repo_path = os.path.join(base_directory, 'data', 'temp', 'repository')
    shutil.copytree(src_data_repo_path, temp_data_repo_path)
    repo = Repository(temp_data_repo_path)
    uuid_to_remove = 'eb0e1bf4-88cf-4cfb-850c-ad11092af8f7'

    repo.remove_dataset(uuid_to_remove)

    assert uuid_to_remove not in repo.metadata['datasets']
    assert not os.path.exists(os.path.join(temp_data_repo_path, 'conf', str(uuid_to_remove) + '.yaml'))
    assert not os.path.exists(os.path.join(temp_data_repo_path, 'datasets', str(uuid_to_remove)))
    repo._valid_repository()
    assert repo.valid_repo == 0
    # ensure that metadata.json has been saved
    repo._load_repository()
    assert uuid_to_remove not in repo.metadata['datasets']

    shutil.rmtree(os.path.join(base_directory, 'data', 'temp'))

def test_remove_dataset_branch(base_directory):
    src_data_repo_path = os.path.join(base_directory, 'data', 'repository')
    os.makedirs(os.path.join(base_directory, 'data', 'temp'), exist_ok=True)
    temp_data_repo_path = os.path.join(base_directory, 'data', 'temp', 'repository')
    shutil.copytree(src_data_repo_path, temp_data_repo_path)
    repo = Repository(temp_data_repo_path)
    uuid_to_remove = '7d8965a5-523c-41e6-8284-8024b7036267'
    expected_removed_uuid = ['7d8965a5-523c-41e6-8284-8024b7036267',
                             '3b96d5a7-0767-41b9-962e-ea4de5d56827',
                             'eb0e1bf4-88cf-4cfb-850c-ad11092af8f7']

    repo.remove_dataset(uuid_to_remove)

    for uuid in expected_removed_uuid:
        assert uuid not in repo.metadata['datasets']
        assert not os.path.exists(os.path.join(temp_data_repo_path, 'conf', str(uuid) + '.yaml'))
        assert not os.path.exists(os.path.join(temp_data_repo_path, 'datasets', str(uuid)))

    repo._valid_repository()
    assert repo.valid_repo == 0
    # ensure that metadata.json has been saved
    repo._load_repository()
    assert set(expected_removed_uuid).isdisjoint(set(repo.metadata['datasets']))

    shutil.rmtree(os.path.join(base_directory, 'data', 'temp'))

def test_get_list_tree_chain(base_directory):
    data_repo_path = os.path.join(base_directory, 'data', 'repository')
    repo = Repository(data_repo_path)

    expected_list = [('8b663706-ab51-4a9a-9a66-eb9ac2c135f3',  None),
                     ('7d8965a5-523c-41e6-8284-8024b7036267', '8b663706-ab51-4a9a-9a66-eb9ac2c135f3'),
                     ('3b96d5a7-0767-41b9-962e-ea4de5d56827', '7d8965a5-523c-41e6-8284-8024b7036267'),
                     ('eb0e1bf4-88cf-4cfb-850c-ad11092af8f7', '3b96d5a7-0767-41b9-962e-ea4de5d56827')]

    result = repo.get_list_tree_chain()

    assert result == expected_list