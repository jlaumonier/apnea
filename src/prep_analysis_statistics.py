import os
import hydra
from codecarbon import EmissionsTracker  # see https://github.com/mlco2/codecarbon/issues/244

from src.data.utils import get_nb_events
from src.pipeline.task import Task


@hydra.main(config_path="../conf", config_name="data-pipeline-statistics", version_base=None)
def main(conf):
    conf.pipeline.data.dataset.source = '70f49f5a-63e0-4bfd-97c4-dfcb3d15ae1b'

    data_repo_path = os.path.join('..', 'data', 'repository')
    simple_task = Task(data_repo_path, conf)
    simple_task.run(conf)

if __name__ == "__main__":
    with EmissionsTracker(output_dir='..', log_level='error') as tracker:
        main()