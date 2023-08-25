from tqdm import tqdm
import hydra
from codecarbon import EmissionsTracker  # see https://github.com/mlco2/codecarbon/issues/244

import pandas as pd
from src.data.datasets.raw_oscar_dataset import RawOscarDataset
from src.data.preparation_tasks import generate_all_rolling_window


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(conf):
    oscar_dataset = RawOscarDataset(output_type='dataframe', limits=None)

   # pd.set_option('display.max_columns', None)
    #print(oscar_dataset[62])

    for idx, df in tqdm(enumerate(oscar_dataset), total=len(oscar_dataset)):
        if df['FlowRate'].isnull().values.any():
            print('ERROR in Flowrate', idx, oscar_dataset.list_files[idx]['label'])
        if df['ApneaEvent'].isnull().values.any():
            print('ERROR in ApneaEvent', idx)


if __name__ == "__main__":
    with EmissionsTracker(output_dir='..', log_level='warning') as tracker:
        main()
