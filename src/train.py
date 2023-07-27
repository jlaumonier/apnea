from functools import partial

import hydra
import torch
import torch.nn as nn
import torch.optim as optim
from poutyne.framework import Model
from torch.utils.data import DataLoader, WeightedRandomSampler
from codecarbon import EmissionsTracker  # see https://github.com/mlco2/codecarbon/issues/244

from src.data.datasets.processed_dataset import ProcessedDataset
from src.data.ts_collator import TSCollator
from src.metrics.accuracy import accuracy
from src.models.basic_lstm_model import BasicLSTMModel


# Thanks to https://discuss.pytorch.org/t/how-to-handle-imbalanced-classes/11264/2
def calculate_weights_dataset_balancing(dataset):
    samples_weight = [1.0] * len(dataset)
    samples_weight_tensor = torch.FloatTensor(samples_weight)
    #samples_weight_tensor = samples_weight_tensor.view(1, len(dataset))
    sampler = WeightedRandomSampler(samples_weight_tensor, len(dataset))
    return sampler

@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(conf):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    batch_size = 32

    processed_dataset_train = ProcessedDataset(output_type='numpy', limits=2000)
    sampler_train = calculate_weights_dataset_balancing(processed_dataset_train)
    processed_dataset_valid = ProcessedDataset(output_type='numpy', limits=1000)
    processed_dataset_test = ProcessedDataset(output_type='numpy', limits=100)

    col = TSCollator()

    # https://www.scottcondron.com/jupyter/visualisation/audio/2020/12/02/dataloaders-samplers-collate.html
    train_loader = DataLoader(dataset=processed_dataset_train, batch_size=batch_size,
                              collate_fn=col.collate_batch, sampler=sampler_train)
    valid_loader = DataLoader(dataset=processed_dataset_valid, batch_size=batch_size, shuffle=True,
                              collate_fn=col.collate_batch)

    model = BasicLSTMModel()
    optimizer = optim.Adam(model.parameters())

    #weights = [496000/500000]
    #class_weights = torch.FloatTensor(weights).cpu()
    loss_fn = nn.MSELoss()

    accuracy_fn = partial(accuracy, device=device)

    model = Model(model, optimizer, loss_fn, batch_metrics=[accuracy_fn], device=device)

    model.fit_generator(train_loader, valid_loader, epochs=10)

    model.evaluate_dataset(processed_dataset_test, batch_size=batch_size, collate_fn=col.collate_batch)

    # TODO test https://www.kaggle.com/code/omershect/learning-pytorch-lstm-deep-learning-with-m5-data


if __name__ == "__main__":
    with EmissionsTracker(output_dir='..', log_level='error') as tracker:
        main()
