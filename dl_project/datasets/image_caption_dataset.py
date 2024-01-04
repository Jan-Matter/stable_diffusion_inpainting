"""Module containing the Dataset for image captioning tasks."""
import io
import multiprocessing
import urllib
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from os import path
from pathlib import Path

import PIL.Image
from datasets import load_dataset
from datasets.utils.file_utils import get_datasets_user_agent
from torchvision.datasets import LSUN, CelebA

USER_AGENT = get_datasets_user_agent()


class ImageCaptionDataset:
    def __init__(self, configs, training=True):
        self.configs = configs
        self.training = training
        if self.configs["name"] == "conceptual_captions":
            self.dataset = load_dataset("conceptual_captions")
            self.dataset = self.dataset.map(
                self.__fetch_images,
                batched=True,
                batch_size=100,
                fn_kwargs={"num_threads": multiprocessing.cpu_count()},
            )
            self.dataset.with_format("torch")
        elif self.configs["name"] == "celeba":
            self.dataset = CelebA("data/celeba", download=True, split="train")
        elif self.configs["name"] == "lsun_church":
            self.dataset = LSUN(
                path.abspath("data/lsun_church"), classes=["church_outdoor_val"]
            )
        else:
            raise ValueError(f'Invalid dataset name {self.configs["name"]}')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __fetch_images(self, batch, num_threads, timeout=None, retries=0):
        fetch_single_image_with_args = partial(
            self.__fetch_single_image, timeout=timeout, retries=retries
        )

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            batch["image"] = list(
                executor.map(fetch_single_image_with_args, batch["image_url"])
            )

        return batch

    def __fetch_single_image(self, image_url, timeout=None, retries=0):
        for _ in range(retries + 1):
            try:
                request = urllib.request.Request(
                    image_url,
                    data=None,
                    headers={"user-agent": USER_AGENT},
                )
                with urllib.request.urlopen(request, timeout=timeout) as req:
                    image = PIL.Image.open(io.BytesIO(req.read()))
                break
            except Exception:
                image = None
        return image


if __name__ == "__main__":
    configs = {"name": "lsun_church"}
    dataset = ImageCaptionDataset(configs)
    print(dataset[0])
