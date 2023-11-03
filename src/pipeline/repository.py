import os#
import json

class Repository:

    def _create_repo(self):
        os.makedirs(self.path)
        metadata = {'datasets': []}
        json.dump(metadata, open(os.path.join(self.path, 'metadata_db.json'), "w"))

    def _load_repository(self):
        metdata = json.load(open(os.path.join(self.path, 'metadata_db.json'), "r"))
        return metdata

    def _valid_repository(self):
        result = True
        for id in self.metadata['datasets']:
            if not os.path.exists(os.path.join(self.path, 'datasets', id)):
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

