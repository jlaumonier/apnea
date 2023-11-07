import os
import shutil

from src.pipeline.repository import Repository
from src.data.datasets.raw_oscar_dataset import RawOscarDataset
def test_repository_init_create_repo():
    os.makedirs('../data/temp', exist_ok=True)

    data_repo_path = '../data/temp/repo'

    assert not os.path.exists(data_repo_path)

    repo = Repository(data_repo_path)

    assert repo.path == data_repo_path
    assert os.path.isdir(data_repo_path)
    assert os.path.isfile(os.path.join(data_repo_path, 'metadata_db.json'))
    assert repo.valid_repo == True

    shutil.rmtree('../data/temp')

def test_repository_init_repo_loaded():
    data_repo_path = '../data/repository'

    repo = Repository(data_repo_path)

    assert repo.valid_repo == True
    assert '8b663706-ab51-4a9a-9a66-eb9ac2c135f3' in repo.metadata['datasets'].keys()

def test_repository_boostrap_copy():
    os.makedirs('../data/temp', exist_ok=True)

    data_repo_path = '../data/temp/repository'
    source_dataset_path = '../data/raw/'

    repo = Repository(data_repo_path)
    guid = repo.bootstrap(source_dataset_path, RawOscarDataset)

    assert str(guid) in repo.metadata['datasets'].keys()
    assert repo.metadata['datasets'][str(guid)]['type'] == 'src.data.datasets.raw_oscar_dataset.RawOscarDataset'
    assert os.path.exists(os.path.join(data_repo_path, 'datasets', str(guid)))
    assert os.path.isfile(os.path.join(data_repo_path, 'conf', str(guid)+'.yaml'))

    shutil.rmtree('../data/temp')

def test_load_repository():
    data_repo_path = '../data/repository'

    repo = Repository(data_repo_path)
    dataset = repo.load_dataset('8b663706-ab51-4a9a-9a66-eb9ac2c135f3')

    assert dataset.__name__ ==  'RawOscarDataset'