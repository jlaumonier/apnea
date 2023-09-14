import hydra
from codecarbon import EmissionsTracker  # see https://github.com/mlco2/codecarbon/issues/244
import numpy as np

from src.data.datasets.pickle_dataset import PickleDataset
from src.data.preparation_tasks import split_dataset


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(conf):
    processed_dataset_complet = PickleDataset(output_type='numpy', src_data_path='../data/processing/pickle/')
    split_dataset(processed_dataset_complet,
                  output_dir_path='../data/processing/split',
                  train_ratio=conf['training']['split']['train_ratio'],
                  valid_ratio=conf['training']['split']['valid_ratio']
                  )

if __name__ == "__main__":
    with EmissionsTracker(output_dir='..', log_level='error') as tracker:
        main()
