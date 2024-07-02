import os
import hydra
from codecarbon import EmissionsTracker  # see https://github.com/mlco2/codecarbon/issues/244

from src.pipeline.repository import Repository
from pyapnea.pytorch.raw_oscar_dataset import RawOscarDataset


@hydra.main(config_path="../conf", config_name="data-pipeline-bootstrap", version_base=None)
def main(conf):
    data_repo_path = os.path.join('..', 'data', 'repository')
    source_dataset_path = os.path.join('..', 'data', 'raw')

    repo = Repository(data_repo_path)
    _ = repo.bootstrap(source_dataset_path, RawOscarDataset,  file_format='raw')


if __name__ == "__main__":
    with EmissionsTracker(output_dir='..', log_level='error') as tracker:
        main()