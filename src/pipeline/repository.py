import os  #
import json
import shutil
import uuid
from typing import List, Tuple, Optional

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
        @param data_repository_path: the path of the data repository
        """
        # data repository path
        self.path = data_repository_path

        if not os.path.isdir(self.path):
            self._create_repo()

        # loading repository database
        self.metadata = self._load_repository()

        self.valid_repo = self._valid_repository()

    def bootstrap(self, source_dataset_path: str, source_dataset_type: type, file_format: str) -> uuid.UUID:
        """
        Bootstrop an empty repository with a given dataset

        @param source_dataset_path: the path of the source dataset to initialize the repository with
        @param source_dataset_type: type of the dataset (see src.data.dataset.*)
        @param file_format: the format of the dataset files
        @return: the UUID of the dataset create by the bootstrap
        """
        new_uuid = uuid.uuid4()
        dest_data_path = os.path.join(self.path, 'datasets', str(new_uuid))
        shutil.copytree(source_dataset_path, dest_data_path)

        self.commit_dataset(new_uuid, source_dataset_type, file_format=file_format, task_config=None)

        return new_uuid

    def load_dataset(self, id: str, getitem_type: str, sub_dataset: str = None, args=None) -> Dataset:
        """
        Load a dataset from the repository

        @param id: the id of the dataset to load
        @param getitem_type: the type of the getitem of the dataset (dataframe or numpy)
        @param sub_dataset: TO BE DETERMINED
        @param args: optional arguments to pass to the instance of the dataset

        @return: loaded the dataset
        """
        cfg = OmegaConf.load(os.path.join(self.path, 'conf', id + '.yaml'))
        cfg['data_path'] = os.path.join(self.path, 'datasets', id)
        if sub_dataset is not None:
            cfg['data_path'] = os.path.join(cfg['data_path'], sub_dataset)
        cfg['getitem_type'] = getitem_type
        if args is not None:
            optional_cfg = OmegaConf.create(args)
            cfg = OmegaConf.merge(cfg, optional_cfg)
        dataset = instantiate(cfg)
        return dataset

    def create_dataset(self) -> (uuid.UUID, str):
        """
        Create a new dataset in repository
        @retuns: the UUID of the created dataset
        """
        new_uuid = uuid.uuid4()
        dest_data_path = os.path.join(self.path, 'datasets', str(new_uuid))

        return new_uuid, dest_data_path

    def commit_dataset(self, guid: uuid.UUID, dataset_type: type, file_format: str,
                       task_config: Optional[DictConfig]) -> None:
        """
        Commit the configurations of the new dataset to the repository
        :param guid: guid of the dataset
        :param dataset_type: type of the dataset (see src.data.dataset.*)
        :param task_config: configuration of the task used to create the dataset. None if the dataset is bootstraped
        """
        self.metadata['datasets'][str(guid)] = {'type': str(dataset_type.__module__) + '.' +
                                                        str(dataset_type.__name__),
                                                'file_format': file_format}
        if task_config is not None:
            self.metadata['datasets'][str(guid)]['task_config'] = OmegaConf.to_container(task_config)
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

    def remove_dataset(self, guid: str) -> None:
        """
        Remove the dataset from the repository and all datasets under it
        :param guid: the guid of the dataset to remove
        """
        # get all uuid to removed
        list_uuids = []
        nodes = self.get_list_tree_chain()
        fifo = [guid]
        while len(fifo) > 0:
            uuid = fifo.pop(0)
            list_uuids.append(uuid)
            childs = [n[0] for n in nodes if n[1] == uuid]
            fifo.extend(childs)
        # remove all elements
        for guid in list_uuids:
            self.metadata['datasets'].pop(guid)
            shutil.rmtree(os.path.join(self.path, 'datasets', guid))
            os.remove(os.path.join(self.path, 'conf', str(guid) + '.yaml'))
            json.dump(self.metadata, open(os.path.join(self.path, 'metadata_db.json'), "w"))

    def get_list_tree_chain(self) -> List[Tuple]:
        """ get all datasets as a list containing (id, id_parent)"""
        list_node = []
        for ds_id in self.metadata['datasets'].keys():
            node = None
            if 'task_config' in self.metadata['datasets'][ds_id]:
                src_id = self.metadata['datasets'][ds_id]['task_config']['pipeline']['data']['dataset']['source']
                node = (ds_id, src_id)
            else:
                node = (ds_id, None)
            list_node.append(node)

        return list_node
