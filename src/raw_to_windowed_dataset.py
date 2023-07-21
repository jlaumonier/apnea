import hydra
from codecarbon import EmissionsTracker  # see https://github.com/mlco2/codecarbon/issues/244

from src.data.datasets.raw_oscar_dataset import RawOscarDataset
from src.data.preparation_tasks import generate_all_rolling_window


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(conf):
    oscar_dataset = RawOscarDataset(output_type='dataframe', limits=1)
    generate_all_rolling_window(oscar_dataset=oscar_dataset,
                                length=500,
                                keep_last_incomplete=False,
                                output_dir_path='../data/processing/windowed/')


if __name__ == "__main__":
    with EmissionsTracker(output_dir='..') as tracker:
        main()
