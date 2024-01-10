import os
from functools import partial
from time import time

import hydra
from hydra.utils import get_original_cwd, instantiate
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from codecarbon import EmissionsTracker  # see https://github.com/mlco2/codecarbon/issues/244
from poutyne import MLFlowLogger
from poutyne.framework import Experiment
from omegaconf import OmegaConf
from torch.utils.data import Dataset


from src.pipeline.repository import Repository
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


def load_split_dataset(id: str, output_type: str, set_type: str, data_repo_path:str) -> Dataset:
    cfg = OmegaConf.load(os.path.join(data_repo_path, 'conf', id + '.yaml'))
    cfg['data_path'] = os.path.join(data_repo_path, 'datasets', id, set_type)
    cfg['output_type']  = output_type
    dataset = instantiate(cfg)
    return dataset


@hydra.main(config_path="../conf", config_name="training-pipeline", version_base=None)
def main(conf):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    batch_size = conf['pipeline']['training']['batch_size']

    data_repo_path = os.path.join('data', 'repository')
    id_split_dataset = conf['pipeline']['training']['dataset']['source']

    # TODO changer pour utiliser repo.load_dataset(sub_dataset)
    processed_dataset_train = load_split_dataset(id_split_dataset, 'numpy', 'train', data_repo_path)
    processed_dataset_valid = load_split_dataset(id_split_dataset, 'numpy', 'valid', data_repo_path)
    processed_dataset_test = load_split_dataset(id_split_dataset, 'numpy', 'test', data_repo_path)

    sampler_train = None
    if conf.pipeline.training.balancing.balancing:
        sampler_train = calculate_weights_dataset_balancing(processed_dataset_train)
        #sampler_valid = calculate_weights_dataset_balancing(processed_dataset_valid)

    # https://www.scottcondron.com/jupyter/visualisation/audio/2020/12/02/dataloaders-samplers-collate.html
    train_loader = DataLoader(dataset=processed_dataset_train,
                              batch_size=batch_size,
                              sampler=sampler_train
                              )
    valid_loader = DataLoader(dataset=processed_dataset_valid,
                              batch_size=batch_size,
                              #sampler=sampler_valid
                              )
    test_loader = DataLoader(dataset=processed_dataset_test,
                             batch_size=batch_size,
                             #collate_fn=col.collate_batch
                             )

    model = BasicLSTMModel()
    optimizer = optim.Adam(model.parameters(), lr=conf['pipeline']['training']['initial_learning_rate'])

    loss_fn = hydra.utils.instantiate(conf['pipeline']['training']['loss'])

    accuracy_fn = partial(accuracy, device=device)

    num_epoch = conf['pipeline']['training']['num_epochs']

    mlflow_logger = MLFlowLogger(experiment_name="experiment",
                                 tracking_uri=conf['global']["logs"]["logger"]['tracking_uri'],
                                 batch_granularity=True)
    mlflow_logger.log_config_params(config_params=conf)  # logging the config dictionary

    hydra_output_path = hydra.core.hydra_config.HydraConfig.get()['runtime']['output_dir']
    working_directory = os.path.join(os.getcwd(),
                                     hydra_output_path, conf['global']["logs"]["local"]['log_dir'],
                                     conf['global']["logs"]["local"]['saving_dir'])

    exp = Experiment(directory=working_directory,
                     network=model,
                     device=device,
                     logging=conf['global']['logs']['local']['logging'],
                     optimizer=optimizer,
                     loss_function=loss_fn,
                     batch_metrics=[accuracy_fn])

    exp.train(train_generator=train_loader,
              valid_generator=valid_loader,
              epochs=num_epoch,
              callbacks=[mlflow_logger])

    exp.test(test_loader)

    print('confmat on valid')
    valid_conf_mat = conf_mat(valid_loader, model, device)

    print('confmat on test')
    test_conf_mat = conf_mat(test_loader, model, device)


    # TODO test https://www.kaggle.com/code/omershect/learning-pytorch-lstm-deep-learning-with-m5-data


if __name__ == "__main__":
    with EmissionsTracker(output_dir='..', log_level='error') as tracker:
        main()
