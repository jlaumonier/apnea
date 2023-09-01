from src.data.datasets.processed_dataset import ProcessedDataset

def test___getitem__():
    ds = ProcessedDataset(data_path='../../data/processed')