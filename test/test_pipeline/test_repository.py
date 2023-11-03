import os
import shutil

from src.pipeline.repository import Repository
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

def test_task_init_repo_loaded():
    data_repo_path = '../data/repository'

    repo = Repository(data_repo_path)

    assert repo.valid_repo == True
    assert '00010203-0405-0607-0809-0a0b0c0d0e0f' in repo.metadata['datasets']