from datasets import load_dataset
from src.datasets.base_dataset import BaseDataset


class HFDataset(BaseDataset):
    def __init__(self, dataset_name, split, *args, **kwargs):
        index = load_dataset(dataset_name, split=split)
        super().__init__(index, *args, **kwargs)
