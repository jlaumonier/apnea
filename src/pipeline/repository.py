import os  #
import json
import shutil
import uuid

from hydra import initialize_config_dir, compose
from hydra.utils import get_original_cwd, instantiate
from omegaconf import OmegaConf, DictConfig
from torch.utils.data import Dataset


class Repository:

    def _create_repo(self):
        os.makedirs(self.path)
        os.makedirs(os.path.join(self.path, 'conf'), exist_ok=True)
        os.makedirs(os.path.join(self.path, 'datasets'), exist_ok=True)
        metadata = {'datasets': {}}
        json.dump(metadata, open(os.path.join(self.path, 'metadata_db.json'), "w"))

    def _load_repository(self):
        metdata = json.load(open(os.path.join(self.path, 'metadata_db.json'), "r"))
        return metdata

    def _valid_repository(self):
        result = 0
        for id in self.metadata['datasets'].keys():
            data_path = os.path.join(self.path, 'datasets', id)
            if not os.path.exists(data_path):
                result = 1
            conf_path = os.path.join(self.path, 'conf', id + '.yaml')
            if not os.path.exists(conf_path):
                result = 2
        nb_keys = len(self.metadata['datasets'].keys())
        nb_conf = len(os.listdir(os.path.join(self.path, 'conf')))
        nb_ds = len(os.listdir(os.path.join(self.path, 'datasets')))
        if nb_keys != nb_conf:
            result = 3
        if nb_keys != nb_ds:
            result = 4
        return result

    def __init__(self, data_repository_path: str):
        """
        @param
        """
        # data repository path
        self.path = data_repository_path

        if not os.path.isdir(self.path):
            self._create_repo()

        # loading repository database
        self.metadata = self._load_repository()

        self.valid_repo = self._valid_repository()

    def bootstrap(self, source_dataset_path: str, source_dataset_type: type) -> uuid.UUID:
        new_uuid = uuid.uuid4()
        self.metadata['datasets'][str(new_uuid)] = {'type': str(source_dataset_type.__module__) + '.' +
                                                            str(source_dataset_type.__name__)}

        dest_data_path = os.path.join(self.path, 'datasets', str(new_uuid))
        shutil.copytree(source_dataset_path, dest_data_path)

        json.dump(self.metadata, open(os.path.join(self.path, 'metadata_db.json'), "w"))

        # configuration
        conf = OmegaConf.create()
        conf['_target_'] = str(source_dataset_type.__module__) + '.' + str(source_dataset_type.__name__)
        conf['data_path'] = '??'

        os.makedirs(os.path.join(self.path, 'conf'), exist_ok=True)
        with open(os.path.join(self.path, 'conf', str(new_uuid) + '.yaml'), "w") as fp:
            OmegaConf.save(config=conf, f=fp)

        # revalidate repository
        self.valid_repo = self._valid_repository()

        return new_uuid

    def load_dataset(self, id: str) -> Dataset:
        cfg = OmegaConf.load(os.path.join(self.path, 'conf', id + '.yaml'))
        cfg['data_path'] = os.path.join(self.path, 'datasets', id)
        dataset = instantiate(cfg)
        return dataset

    def create_dataset(self) -> (uuid.UUID, str):
        new_uuid = uuid.uuid4()
        dest_data_path = os.path.join(self.path, 'datasets', str(new_uuid))

        return new_uuid, dest_data_path

    def commit_dataset(self, guid: uuid.UUID, dataset_type: type, conf: DictConfig):
        # TODO code trop recopié de boostrap
        self.metadata['datasets'][str(guid)] = {'type': str(dataset_type.__module__) + '.' +
                                                        str(dataset_type.__name__),
                                                'task_config': OmegaConf.to_container(conf)}
        json.dump(self.metadata, open(os.path.join(self.path, 'metadata_db.json'), "w"))

        # configuration
        conf = OmegaConf.create()
        conf['_target_'] = str(dataset_type.__module__) + '.' + str(dataset_type.__name__)
        conf['data_path'] = '??'

        os.makedirs(os.path.join(self.path, 'conf'), exist_ok=True)
        with open(os.path.join(self.path, 'conf', str(guid) + '.yaml'), "w") as fp:
            OmegaConf.save(config=conf, f=fp)

        # revalidate repository
        self.valid_repo = self._valid_repository()
