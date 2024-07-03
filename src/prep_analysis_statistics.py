import os
import hydra
from codecarbon import EmissionsTracker  # see https://github.com/mlco2/codecarbon/issues/244

from src.data.statistics import task_generate_statistics
from src.pipeline.repository import Repository


@hydra.main(config_path="../conf", config_name="data-pipeline-statistics", version_base=None)
def main(conf):
    data_src_id = 'e44f9d68-72e2-4fb4-866f-0dcab7b04917'

    data_repo_path = os.path.join('..', 'data', 'repository')
    outputs_statistics_path = os.path.join('..', 'outputs', 'statistics')
    repo = Repository(data_repo_path)
    dataset = repo.load_dataset(data_src_id, 'dataframe')
    task_generate_statistics(dataset, outputs_statistics_path)

if __name__ == "__main__":
    with EmissionsTracker(output_dir='..', log_level='error') as tracker:
        main()