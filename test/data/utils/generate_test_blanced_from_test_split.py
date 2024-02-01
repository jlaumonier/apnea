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
        cfg = compose(config_name="data-pipeline-balancing")
        cfg.pipeline.data.dataset.source = '42d374a5-5644-479a-89f0-651b413dd275'
        cfg.pipeline.data.dataset.sub_src = ['train']

        task_split = Task(src_data_repo_path, cfg)
        task_split.run(cfg)


if __name__ == '__main__':
    main()