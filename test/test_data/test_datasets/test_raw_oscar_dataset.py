from src.data.datasets.raw_oscar_dataset import RawOscarDataset

def test___getitem__():

    ds = RawOscarDataset(data_path='../../data/raw')

    assert len(ds) == 1
    # id elmnt, inputs, first timestep, first sensor
    assert ds[0][0][0][0] == -38.76000154018402
    # id elmnt, class
    assert ds[0][1][0] == 0