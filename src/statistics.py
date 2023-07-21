import hydra
from tqdm import tqdm
from codecarbon import EmissionsTracker  # see https://github.com/mlco2/codecarbon/issues/244

from src.data.datasets.processed_dataset import ProcessedDataset


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(conf):

    processed_dataset = ProcessedDataset(output_type='dataframe', limits=1000)

    stats = {'nb_files': len(processed_dataset)}

    # get the existence of event in the file
    nb_contain_event = 0
    with tqdm(total=len(processed_dataset), position=0, colour='red', ncols=80) as pbar:
        for item in processed_dataset:
            nb_contain_event += (1.0 in item['Obstructive'].values)
            pbar.update(1)

    stats['nb_contain_event'] = nb_contain_event
    print(stats)

if __name__ == "__main__":
    with EmissionsTracker(output_dir='..', log_level='error') as tracker:
        main()