import uuid
import os
from src.pipeline.task import Task
from torch.utils.data import Dataset

def test_task_init():
    data_repo_path = '../data/repository'

    simple_task = Task(data_repo_path, '00010203-0405-0607-0809-0a0b0c0d0e0f')

    assert simple_task.src_id == uuid.UUID('00010203-0405-0607-0809-0a0b0c0d0e0f')
    assert type(simple_task.src_dataset) == Dataset
    assert type(simple_task.dest_id) == uuid.UUID

