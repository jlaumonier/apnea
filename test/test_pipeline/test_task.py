import uuid
import os
import shutil
import pytest

from torch.utils.data import Dataset
from hydra import initialize, compose
from pyapnea.pytorch.raw_oscar_dataset import RawOscarDataset

from src.pipeline.repository import Repository
from src.pipeline.task import Task


@pytest.fixture(scope="function")
def relative_path():
    yield '../'


def test_task_init(base_directory, relative_path):
    with initialize(version_base=None, config_path=os.path.join(relative_path, 'conf')):
        # config is relative to a module
        cfg = compose(config_name="data-pipeline-windowed")
        cfg.pipeline.data.dataset.source = '8b663706-ab51-4a9a-9a66-eb9ac2c135f3'

        data_repo_path = os.path.join(base_directory, 'data', 'repository')

        simple_task = Task(data_repo_path, cfg)

        assert simple_task.src_id == uuid.UUID('8b663706-ab51-4a9a-9a66-eb9ac2c135f3')
        assert isinstance(simple_task.src_dataset, list)
        assert isinstance(simple_task.src_dataset[0], tuple)
        assert isinstance(simple_task.src_dataset[0][1], Dataset)
        assert simple_task.dest_id is None


def test_task_run(base_directory, relative_path):
    with initialize(version_base=None, config_path=os.path.join(relative_path, 'conf')):
        # config is relative to a module
        cfg = compose(config_name="data-pipeline-windowed")

        # create a temp repo
        os.makedirs(os.path.join(base_directory, 'data', 'temp'), exist_ok=True)
        data_repo_path = os.path.join(base_directory, 'data', 'temp', 'repo')
        source_dataset_path = os.path.join(base_directory, 'data', 'raw')
        repo = Repository(data_repo_path)
        guid = repo.bootstrap(source_dataset_path, RawOscarDataset, 'raw')
        cfg.pipeline.data.dataset.source = str(guid)

        simple_task = Task(data_repo_path, cfg)
        simple_task.run(cfg)

        assert type(simple_task.dest_id) is uuid.UUID
        assert simple_task.repo.valid_repo == 0

        shutil.rmtree(os.path.join(base_directory, 'data', 'temp'))



