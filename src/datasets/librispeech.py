import json
import os
import shutil
import tarfile
from logging import Logger
from pathlib import Path
from typing import Optional

import wget

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH

LIBRISPEECH_DATASET_URL = "https://www.openslr.org/resources/12/{}.tar.gz"


class LibrispeechDatasetSplit:
    def __init__(
        self,
        split: str,
        logger: Logger,
        dataset_dir: Optional[str] = None,
        save_meta_info: bool = False,
    ):
        self.split = split
        self.split_dir = Path(dataset_dir) / split
        self.save_meta_info = save_meta_info
        self.logger = logger

    def _download(self):
        tar_path = self.split_dir / f"{self.split}.tar.gz"
        url = LIBRISPEECH_DATASET_URL.format(self.split)
        self.logger.info(f"Split '{self.split}' was not found. Downloading from {url}")
        wget.download(url, str(tar_path))

        self.logger.info("Extracting")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(self.split_dir)
        os.remove(tar_path)

        if self.save_meta_info:
            for file in (self.split_dir / "LibriSpeech").iterdir():
                if file.is_file():
                    shutil.move(file, self.split_dir.parent / file.name)

        for file in (self.split_dir / "LibriSpeech" / self.split).iterdir():
            shutil.move(file, self.split_dir / file.name)
        shutil.rmtree(self.split_dir / "LibriSpeech")

    def _create_index(self):
        index = []
        for path, _, files in os.walk(self.split_dir):
            for file in files:
                if file.endswith(".flac"):
                    index.append({"path": str(self.split_dir / path / file)})
        with open(self.split_dir / "index.json", "w") as f:
            json.dump(index, f, indent=2)
        return index

    def get_index(self):
        self.logger.info(f"Loading split {self.split}")

        if not self.split_dir.exists():
            self.split_dir.mkdir(parents=True, exist_ok=True)
            self._download()

        index_path = self.split_dir / "index.json"
        if not index_path.exists():
            self.logger.info("Creating index")
            self._create_index()

        with open(index_path, "r") as f:
            return json.load(f)


class LibrispeechDataset(BaseDataset):
    def __init__(
        self,
        split: str | list[str],
        logger: Logger,
        dataset_dir: Optional[str] = None,
        *args,
        **kwargs,
    ):
        if dataset_dir is None:
            self.dataset_dir = ROOT_PATH / "data" / "datasets" / "librispeech"
        else:
            self.dataset_dir = Path(dataset_dir)

        if isinstance(split, str):
            split = [split]
        self.splits = [
            LibrispeechDatasetSplit(
                split=split,
                logger=logger,
                dataset_dir=self.dataset_dir,
                save_meta_info=True,
            )
            for split in split
        ]

        index = sum((split.get_index() for split in self.splits), start=[])
        logger.info("Index was successfully loaded")
        super().__init__(index, *args, **kwargs)
