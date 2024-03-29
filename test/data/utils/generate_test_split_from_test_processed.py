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
        cfg = compose(config_name="data-pipeline-split")
        cfg.pipeline.data.dataset.source = 'a67ff056-fd10-4b41-bd1a-104f9e23279e'

        task_split = Task(src_data_repo_path, cfg)
        task_split.run(cfg)


if __name__ == '__main__':
    main()