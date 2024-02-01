from hydra import initialize, compose
import pytest
import os
import shutil

from src.pipeline.task import Task


@pytest.fixture(scope="function")
def relative_path():
    yield './'


def test_task_raw_to_windowed_run(base_directory, relative_path):
    src_data_repo_path = os.path.join(base_directory, 'data', 'repository')
    os.makedirs(os.path.join(base_directory, 'data', 'temp'), exist_ok=True)
    temp_data_repo_path = os.path.join(base_directory, 'data', 'temp', 'repository')
    shutil.copytree(src_data_repo_path, temp_data_repo_path)

    with initialize(version_base=None, config_path=os.path.join(relative_path, 'conf')):
        cfg = compose(config_name="data-pipeline-windowed")
        cfg.pipeline.data.dataset.source = '8b663706-ab51-4a9a-9a66-eb9ac2c135f3'

        task_raw_to_windowed = Task(temp_data_repo_path, cfg)
        task_raw_to_windowed.run(cfg)

    shutil.rmtree(os.path.join(base_directory, 'data', 'temp'))


def test_task_processed_to_pickle(base_directory, relative_path):
    src_data_repo_path = os.path.join(base_directory, 'data', 'repository')
    os.makedirs(os.path.join(base_directory, 'data', 'temp'), exist_ok=True)
    temp_data_repo_path = os.path.join(base_directory, 'data', 'temp', 'repository')
    shutil.copytree(src_data_repo_path, temp_data_repo_path)

    with initialize(version_base=None, config_path=os.path.join(relative_path, 'conf')):
        cfg = compose(config_name="data-pipeline-pickle")
        cfg.pipeline.data.dataset.source = '9e81da40-41a1-4f9b-9bba-41de71b0ebd9'

        task_raw_to_windowed = Task(temp_data_repo_path, cfg)
        task_raw_to_windowed.run(cfg)

    shutil.rmtree(os.path.join(base_directory, 'data', 'temp'))


def test_task_split_to_balanced(base_directory, relative_path):
    src_data_repo_path = os.path.join(base_directory, 'data', 'repository')
    os.makedirs(os.path.join(base_directory, 'data', 'temp'), exist_ok=True)
    temp_data_repo_path = os.path.join(base_directory, 'data', 'temp', 'repository')
    shutil.copytree(src_data_repo_path, temp_data_repo_path)

    with initialize(version_base=None, config_path=os.path.join(relative_path, 'conf')):
        cfg = compose(config_name="data-pipeline-balancing")
        cfg.pipeline.data.dataset.source = '42d374a5-5644-479a-89f0-651b413dd275'
        cfg.pipeline.data.dataset.sub_src = ['train']

        task_balancing = Task(temp_data_repo_path, cfg)
        task_balancing.run(cfg)

        list_files = os.listdir(os.path.join(task_balancing.repo.path, 'datasets', str(task_balancing.dest_id)))
        list_files.sort()
        assert len(list_files) == 3
        assert 'train' in list_files
        assert 'test' in list_files
        assert 'valid' in list_files

        train_ds = task_balancing.repo.load_dataset(str(task_balancing.dest_id),
                                                    cfg.pipeline.data.dataset.getitem_type,
                                                    sub_dataset='train')
        assert len(train_ds) == 10
        test_ds = task_balancing.repo.load_dataset(str(task_balancing.dest_id),
                                                    cfg.pipeline.data.dataset.getitem_type,
                                                    sub_dataset='test')
        assert len(test_ds) == 149
        valid_ds = task_balancing.repo.load_dataset(str(task_balancing.dest_id),
                                                    cfg.pipeline.data.dataset.getitem_type,
                                                    sub_dataset='valid')
        assert len(valid_ds) == 148

    shutil.rmtree(os.path.join(base_directory, 'data', 'temp'))
