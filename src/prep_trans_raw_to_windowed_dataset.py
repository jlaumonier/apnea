import os
import hydra
from codecarbon import EmissionsTracker  # see https://github.com/mlco2/codecarbon/issues/244

from src.pipeline.task import Task


@hydra.main(config_path="../conf", config_name="data-pipeline-rolling", version_base=None)
def main(conf):
    conf.pipeline.data.dataset.source = 'fc0599ca-91cb-4b7c-b5cb-4d4af1d3a731'

    data_repo_path = os.path.join('..', 'data', 'repository')
    simple_task = Task(data_repo_path, conf)
    simple_task.run(conf)


if __name__ == "__main__":
    with EmissionsTracker(output_dir='..', log_level='error') as tracker:
        main()
