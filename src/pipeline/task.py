import os.path
import uuid
from src.pipeline.repository import Repository

class Task:
    """
    Task used for data pipeline
    """

    def __init__(self, data_repository_path: str, src_id: str):
        """
        @param data_repository_path path of the data repository. Create if does not exists
        @param src_id string uuid taht define the source dataset in repository
        """
        self.repo = Repository(data_repository_path)
        # source dataset id
        self.src_id = None
        if src_id in self.repo.metadata['datasets']:
            self.src_id = uuid.UUID(src_id)
            # TODO next dev task : loading dataset according to configuration
            self.src_dataset = None
        # destination dataset id
        self.dest_id = uuid.uuid4()

    def run(self):
        pass

