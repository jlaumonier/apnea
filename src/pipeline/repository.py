import os#
import json
import shutil
import uuid

from hydra import initialize, compose
from hydra.utils import get_original_cwd, instantiate
from torch.utils.data import Dataset

class Repository:

    def _create_repo(self):
        os.makedirs(self.path)
        metadata = {'datasets': {}}
        json.dump(metadata, open(os.path.join(self.path, 'metadata_db.json'), "w"))

    def _load_repository(self):
        metdata = json.load(open(os.path.join(self.path, 'metadata_db.json'), "r"))
        return metdata

    def _valid_repository(self):
        result = True
        for id in self.metadata['datasets'].keys():
            data_path = os.path.join(self.path, 'datasets', id)
            if not os.path.exists(data_path):
                result = False
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
        self.metadata['datasets'][str(new_uuid)] = {}

        dest_data_path = os.path.join(self.path, 'datasets', str(new_uuid))
        shutil.copytree(source_dataset_path, dest_data_path)

        json.dump(self.metadata, open(os.path.join(self.path, 'metadata_db.json'), "w"))

        # revalidate repository
        self.valid_repo = self._valid_repository()

        return new_uuid

    def load_dataset(self, id: str) -> Dataset:
        with initialize(version_base=None, config_path=os.path.join(self.path, 'conf')):
            # config is relative to a module
            cfg = compose(config_name=id)
            dataset = instantiate(cfg)
            return dataset

