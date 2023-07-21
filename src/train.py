from functools import partial

import hydra
import torch
import torch.nn as nn
import torch.optim as optim
from poutyne.framework import Model
from torch.utils.data import DataLoader
from codecarbon import EmissionsTracker  # see https://github.com/mlco2/codecarbon/issues/244

from src.data.datasets.processed_dataset import ProcessedDataset
from src.data.ts_collator import TSCollator
from src.metrics.accuracy import accuracy
from src.models.basic_lstm_model import BasicLSTMModel


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(conf):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    batch_size = 2

    processed_dataset = ProcessedDataset(output_type='numpy', limits=100)

    col = TSCollator()

    # https://www.scottcondron.com/jupyter/visualisation/audio/2020/12/02/dataloaders-samplers-collate.html
    train_loader = DataLoader(dataset=processed_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=col.collate_batch)
    valid_loader = DataLoader(dataset=processed_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=col.collate_batch)

    model = BasicLSTMModel()
    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.CrossEntropyLoss()

    accuracy_fn = partial(accuracy, device=device)

    model = Model(model, optimizer, loss_fn, batch_metrics=[accuracy_fn], device=device)

    model.fit_generator(train_loader, valid_loader, epochs=1)

    model.evaluate_dataset(processed_dataset, batch_size=batch_size, collate_fn=col.collate_batch)

    # TODO test https://www.kaggle.com/code/omershect/learning-pytorch-lstm-deep-learning-with-m5-data


if __name__ == "__main__":
    with EmissionsTracker(output_dir='..') as tracker:
        main()
