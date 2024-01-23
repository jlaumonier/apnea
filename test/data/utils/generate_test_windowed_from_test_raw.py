from hydra import initialize, compose
import pytest
import os
import shutil

from src.pipeline.task import Task

def main():
    relative_path = '../..'
    base_directory = os.path.realpath(os.path.dirname(__file__))
    base_directory = os.path.join(base_directory, relative_path)
    src_data_repo_path = os.path.join(base_directory, 'data', 'repository')

    with initialize(version_base=None, config_path=os.path.join(relative_path, 'conf')):
        cfg = compose(config_name="data-pipeline-windowed")
        cfg.pipeline.data.dataset.source = '8b663706-ab51-4a9a-9a66-eb9ac2c135f3'

        task_raw_to_windowed = Task(src_data_repo_path, cfg)
        task_raw_to_windowed.run(cfg)


if __name__ == '__main__':
    main()