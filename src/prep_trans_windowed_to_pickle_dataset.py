import hydra
from codecarbon import EmissionsTracker  # see https://github.com/mlco2/codecarbon/issues/244

from src.data.datasets.processed_dataset import ProcessedDataset
from src.data.preparation_tasks import generate_pickle_dataset


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(conf):
    oscar_dataset = ProcessedDataset(output_type='numpy', limits=None)

    output_directory = '../data/processing/pickle/'
    generate_pickle_dataset(oscar_dataset, output_directory)

if __name__ == "__main__":
    with EmissionsTracker(output_dir='..', log_level='error') as tracker:
        main()
