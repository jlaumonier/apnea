import os
import hydra
from codecarbon import EmissionsTracker  # see https://github.com/mlco2/codecarbon/issues/244

from src.pipeline.task import Task


@hydra.main(config_path="../conf", config_name="data-pipeline-rolling", version_base=None)
def main(conf):
    conf.pipeline.data.dataset.source = '5d91f31f-b880-41d6-b5fe-fd15ebb15e20'

    data_repo_path = os.path.join('..', 'data', 'repository')
    simple_task = Task(data_repo_path, conf)
    simple_task.run(conf)


if __name__ == "__main__":
    with EmissionsTracker(output_dir='..', log_level='error') as tracker:
        main()
