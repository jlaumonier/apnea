import os.path
import uuid
from hydra.utils import call
import shutil

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
        self.sub_srcs = cfg.pipeline.data.dataset.sub_src
        if  self.sub_srcs is None or self.sub_srcs == '??':
            self.sub_srcs = [None]
        getitem_type = cfg.pipeline.data.dataset.getitem_type
        self.src_id = None
        self.src_dataset = []
        if src_id in self.repo.metadata['datasets']:
            self.src_id = uuid.UUID(src_id)
            for sub_src in self.sub_srcs:
                self.src_dataset.append((sub_src, self.repo.load_dataset(src_id, getitem_type, sub_dataset=sub_src)))
        self.dest_id = None

    def run(self, cfg):
        guid, dataset_path = self.repo.create_dataset()
        dataset_type = None
        file_format = None
        # TODO os.path.join(self.repo.path, 'datasets', str(self.src_id)) should be given by repo ??
        list_sub_ds_path = os.listdir(os.path.join(self.repo.path, 'datasets', str(self.src_id)))
        for p in list_sub_ds_path:
            if p in self.sub_srcs:
                ds = [src_ds for src_ds in self.src_dataset if src_ds[0]==p][0]
                dataset_type, file_format = call(cfg.pipeline.data.tasks.task_func,
                                                 oscar_dataset=ds[1],
                                                 output_dir_path=os.path.join(dataset_path, ds[0]))
            else:
                shutil.copytree(os.path.join(self.repo.path, 'datasets', str(self.src_id), p),
                                os.path.join(dataset_path, p))
        self.repo.commit_dataset(guid, dataset_type, file_format, task_config=cfg)
        self.dest_id = guid
