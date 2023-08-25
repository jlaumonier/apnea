import hydra
import pandas as pd
import matplotlib.pyplot as plt
from codecarbon import EmissionsTracker  # see https://github.com/mlco2/codecarbon/issues/244

from src.data.datasets.raw_biopoint_dataset import RawBioPointDataset

pd.set_option('display.max_columns', None)
@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(conf):


    biopoint_ds = RawBioPointDataset(output_type='dataframe',
                                     channels=['BioImp', 'IMU', 'TEMPERATURE', 'ECG', 'EMG'],
                                     resampling_freq=40)
    data = biopoint_ds[0]
    print(data.head(20))

    plt.figure(figsize=(20, 10))
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4)
    ax1.plot(data.index, data['BIO_IMPEDANCE'])
    ax2.plot(data.index, data['ECG'])
    ax3.plot(data.index, data['EMG'])
    ax4.plot(data.index, data['Temperature'])
    plt.show()

if __name__ == "__main__":
    with EmissionsTracker(output_dir='..', log_level='error') as tracker:
        main()
