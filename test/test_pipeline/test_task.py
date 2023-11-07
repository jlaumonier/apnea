import uuid

from torch.utils.data import Dataset
from hydra import initialize, compose

from src.pipeline.task import Task

def test_task_init():
    with initialize(version_base=None, config_path="../conf"):
        # config is relative to a module
        cfg = compose(config_name="data-pipeline")

        data_repo_path = '../data/repository'

        simple_task = Task(data_repo_path, '8b663706-ab51-4a9a-9a66-eb9ac2c135f3')

        assert simple_task.src_id == uuid.UUID('8b663706-ab51-4a9a-9a66-eb9ac2c135f3')
        assert type(simple_task.src_dataset) == Dataset
        assert type(simple_task.dest_id) == uuid.UUID

