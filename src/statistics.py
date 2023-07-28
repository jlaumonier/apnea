import hydra
from codecarbon import EmissionsTracker  # see https://github.com/mlco2/codecarbon/issues/244

from src.data.datasets.processed_dataset import ProcessedDataset
from src.data.utils import get_nb_events


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(conf):

    processed_dataset = ProcessedDataset(output_type='dataframe', limits=None)

    stats = {'nb_files': len(processed_dataset)}

    # get the existence of event in the file
    nb_contain_event, events = get_nb_events(processed_dataset)

    stats['nb_contain_event'] = nb_contain_event
    print(stats)

if __name__ == "__main__":
    with EmissionsTracker(output_dir='..', log_level='error') as tracker:
        main()