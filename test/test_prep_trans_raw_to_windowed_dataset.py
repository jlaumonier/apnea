from hydra import initialize, compose
import pytest
import os

from src.pipeline.task import Task

@pytest.fixture(scope="function")
def relative_path():
    yield './'

def test_task_raw_to_windowed_run(base_directory, relative_path):
    with initialize(version_base=None, config_path=os.path.join(relative_path, 'conf')):
        cfg = compose(config_name="data-pipeline")
        cfg.pipeline.data.dataset.source = '8b663706-ab51-4a9a-9a66-eb9ac2c135f3'

        data_repo_path = os.path.join(base_directory, 'data', 'repository')

        task_raw_to_windowed = Task(data_repo_path, cfg)
        task_raw_to_windowed.run(cfg)

