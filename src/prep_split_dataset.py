import os
import hydra
from codecarbon import EmissionsTracker  # see https://github.com/mlco2/codecarbon/issues/244

from src.pipeline.task import Task

@hydra.main(config_path="../conf", config_name="data-pipeline-split", version_base=None)
def main(conf):
    conf.pipeline.data.dataset.source = 'fbe7c01f-589b-4b5b-8a36-554dfc6880b2'

    data_repo_path = os.path.join('data', 'repository')
    simple_task = Task(data_repo_path, conf)
    simple_task.run(conf)

if __name__ == "__main__":
    with EmissionsTracker(output_dir='..', log_level='error') as tracker:
        main()
