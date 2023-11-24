import uuid
import os
import shutil
import pytest

from torch.utils.data import Dataset
from hydra import initialize, compose

from src.pipeline.repository import Repository
from src.pipeline.task import Task
from src.data.datasets.raw_oscar_dataset import RawOscarDataset

@pytest.fixture(scope="function")
def relative_path():
    yield '../'

def test_task_init(base_directory, relative_path):
    with initialize(version_base=None, config_path=os.path.join(relative_path, 'conf')):
        # config is relative to a module
        cfg = compose(config_name="data-pipeline")
        cfg.pipeline.data.dataset.source = '8b663706-ab51-4a9a-9a66-eb9ac2c135f3'

        data_repo_path = os.path.join(base_directory, 'data', 'repository')

        simple_task = Task(data_repo_path, cfg)

        assert simple_task.src_id == uuid.UUID('8b663706-ab51-4a9a-9a66-eb9ac2c135f3')
        assert isinstance(simple_task.src_dataset, Dataset)
        assert type(simple_task.dest_id) == uuid.UUID


def test_task_execute(base_directory, relative_path):
    with initialize(version_base=None, config_path=os.path.join(relative_path, 'conf')):
        # config is relative to a module
        cfg = compose(config_name="data-pipeline")

        # create a temp repo
        os.makedirs(os.path.join(base_directory, 'data', 'temp'), exist_ok=True)
        data_repo_path = os.path.join(base_directory, 'data', 'temp', 'repo')
        source_dataset_path = os.path.join(base_directory, 'data', 'raw')
        repo = Repository(data_repo_path)
        guid = repo.bootstrap(source_dataset_path, RawOscarDataset)
        cfg.pipeline.data.dataset.source = str(guid)

        simple_task = Task(data_repo_path, cfg)
        simple_task.run(cfg)

        assert simple_task.repo.valid_repo == 0

        shutil.rmtree(os.path.join(base_directory, 'data', 'temp'))


