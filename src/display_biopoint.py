import hydra
import matplotlib.pyplot as plt
from codecarbon import EmissionsTracker  # see https://github.com/mlco2/codecarbon/issues/244

from src.data.datasets.raw_biopoint_dataset import RawBioPointDataset


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(conf):
    biopoint_ds = RawBioPointDataset(output_type='dataframe', channels=['BioImp', 'IMU', 'TEMPERATURE'])
    # impossible de fusionner ECG avec les autres
    biopoint_ds2 = RawBioPointDataset(output_type='dataframe', channels=['ECG'])
    biopoint_ds3 = RawBioPointDataset(output_type='dataframe', channels=['EMG'])
    data = biopoint_ds[0]
    data2 = biopoint_ds2[0]
    data3 = biopoint_ds3[0]
    print(data3)

    plt.figure(figsize=(20, 10))
    fig, (ax1, ax2, ax3) = plt.subplots(3)
    ax1.plot(data['Time'], data['BIO_IMPEDANCE'])
    ax2.plot(data2['Time'], data2['ECG'])
    ax3.plot(data3['EMG'])
    plt.show()

if __name__ == "__main__":
    with EmissionsTracker(output_dir='..') as tracker:
        main()
