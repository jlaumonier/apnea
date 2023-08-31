import os
from functools import partial
from time import time

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from codecarbon import EmissionsTracker  # see https://github.com/mlco2/codecarbon/issues/244
from poutyne import MLFlowLogger
from poutyne.framework import Experiment


from src.data.datasets.processed_dataset import ProcessedDataset
from src.data.datasets.pickle_dataset import PickleDataset
from src.data.ts_collator import TSCollator
from src.metrics.accuracy import accuracy
from src.metrics.confusion_matrix import conf_mat
from src.models.basic_lstm_model import BasicLSTMModel
from src.data.utils import is_contain_event, get_nb_events


# Thanks to https://discuss.pytorch.org/t/how-to-handle-imbalanced-classes/11264/2
def calculate_weights_dataset_balancing(dataset):
    print('balancing dataset')
    nb_events, events = get_nb_events(dataset)
    weight_event =  (len(dataset) - nb_events) / len(dataset)
    weight_no_event = nb_events / len(dataset)
    samples_weight = [weight_event if item==1.0 else weight_no_event for item in events ]
    samples_weight_tensor = torch.FloatTensor(samples_weight)
    sampler = WeightedRandomSampler(samples_weight_tensor, len(dataset))
    print('balancing dataset finished')
    return sampler




@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(conf):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    batch_size = 1

    processed_dataset_complet = PickleDataset(output_type='numpy')
    len_complete_dataset = len(processed_dataset_complet)
    # take only a subset of complete dataset
    len_complete_dataset = int(len_complete_dataset * 1.0)
    #percentage_split = (0.8, 0.1, 0.1)
    percentage_split = (1.0, 0, 0)
    cumul_perc_split = np.cumsum(percentage_split)

    idx_tvt_set = [int(i) for i in (cumul_perc_split * len_complete_dataset)]
    print(idx_tvt_set)

    processed_dataset_train = PickleDataset(output_type='numpy', limits=slice(0,idx_tvt_set[0]))
    # sampler_train = calculate_weights_dataset_balancing(processed_dataset_train)
    # processed_dataset_valid = PickleDataset(output_type='numpy', limits=slice(idx_tvt_set[0]+1,idx_tvt_set[1]))
    # sampler_valid = calculate_weights_dataset_balancing(processed_dataset_valid)
    # processed_dataset_test = PickleDataset(output_type='numpy', limits=slice(idx_tvt_set[1]+1,idx_tvt_set[2]))

    col = TSCollator()

    # https://www.scottcondron.com/jupyter/visualisation/audio/2020/12/02/dataloaders-samplers-collate.html
    train_loader = DataLoader(dataset=processed_dataset_train, batch_size=batch_size,
                              collate_fn=col.collate_batch,
                              #sampler=sampler_train
                              )
    # valid_loader = DataLoader(dataset=processed_dataset_valid, batch_size=batch_size,
    #                           collate_fn=col.collate_batch, sampler=sampler_valid)
    # test_loader = DataLoader(dataset=processed_dataset_test,  batch_size=batch_size,
    #                          collate_fn=col.collate_batch)

    model = BasicLSTMModel()
    optimizer = optim.Adam(model.parameters())

    #loss_fn = nn.MSELoss()
    loss_fn = nn.CrossEntropyLoss()

    accuracy_fn = partial(accuracy, device=device)

    num_epoch = 100

    mlflow_logger = MLFlowLogger(experiment_name="experiment",
                                 tracking_uri=conf["logs"]["logger"]['tracking_uri'],
                                 batch_granularity=True)
    mlflow_logger.log_config_params(config_params=conf)  # logging the config dictionary

    # TODo merge with hydra output dir
    working_directory = os.path.join(os.getcwd(), conf["logs"]["local"]['log_dir'], conf["logs"]["local"]['saving_dir'])

    exp = Experiment(directory=working_directory,
                     network=model,
                     device=device,
                     logging=conf['logs']['local']['logging'],
                     optimizer=optimizer,
                     loss_function=loss_fn,
                     batch_metrics=[accuracy_fn])

    exp.train(train_generator=train_loader,
              valid_generator=train_loader,
              epochs=num_epoch,
              callbacks=[mlflow_logger])

    exp.test(train_loader)

    print('confmat on valid')
    conf_mat(train_loader, model, device)

    print('confmat on test')
    conf_mat(train_loader, model, device)

    # TODO test https://www.kaggle.com/code/omershect/learning-pytorch-lstm-deep-learning-with-m5-data


if __name__ == "__main__":
    with EmissionsTracker(output_dir='..', log_level='error') as tracker:
        main()
