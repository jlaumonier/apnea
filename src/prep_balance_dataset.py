import os
import hydra
from codecarbon import EmissionsTracker  # see https://github.com/mlco2/codecarbon/issues/244

from src.pipeline.task import Task

@hydra.main(config_path="../conf", config_name="data-pipeline-balancing", version_base=None)
def main(conf):
    conf.pipeline.data.dataset.source = 'db194b54-ab94-4ed0-9973-2ca5fbeeb8f6'
    conf.pipeline.data.dataset.sub_src = ['train']

    data_repo_path = os.path.join('data', 'repository')
    simple_task = Task(data_repo_path, conf)
    simple_task.run(conf)

if __name__ == "__main__":
    with EmissionsTracker(output_dir='..', log_level='error') as tracker:
        main()
