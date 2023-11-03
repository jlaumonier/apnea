from src.pipeline.task import Task

def test_task_raw_to_windowed_run():
    data_repo_path = 'data/repository'
    task_raw_to_windowed = Task(data_repo_path, '00010203-0405-0607-0809-0a0b0c0d0e0f' )
    task_raw_to_windowed.run()

