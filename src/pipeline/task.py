import os.path
import uuid
from hydra.utils import call

from src.pipeline.repository import Repository
from src.data.datasets.processed_dataset import ProcessedDataset


class Task:
    """
    Task used for data pipeline
    """

    def __init__(self, data_repository_path: str, cfg):
        """
        @param data_repository_path path of the data repository. Create if does not exists
        @param src_id string uuid taht define the source dataset in repository
        """
        self.repo = Repository(data_repository_path)
        # source dataset id
        src_id = cfg.pipeline.data.dataset.source
        sub_srcs = cfg.pipeline.data.dataset.sub_src
        if sub_srcs is None or sub_srcs == '??':
            sub_srcs = [None]
        getitem_type = cfg.pipeline.data.dataset.getitem_type
        self.src_id = None
        self.src_dataset = []
        if src_id in self.repo.metadata['datasets']:
            self.src_id = uuid.UUID(src_id)
            for sub_src in sub_srcs:
                self.src_dataset.append((sub_src, self.repo.load_dataset(src_id, getitem_type, sub_dataset=sub_src)))
        # destination dataset id
        self.dest_id = uuid.uuid4()

    def run(self, cfg):
        guid, dataset_path = self.repo.create_dataset()
        dataset_type, file_format = call(cfg.pipeline.data.tasks.task_func,
                                         oscar_dataset=self.src_dataset[0][1],
                                         output_dir_path=dataset_path)
        self.repo.commit_dataset(guid, dataset_type, file_format, task_config=cfg)
