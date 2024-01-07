"""Module containing the Dataset for image captioning tasks."""
import tensorflow as tf
import tensorflow_datasets as tfds
import io
import multiprocessing
import urllib
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from os import path
from pathlib import Path
import numpy as np
import torch

import PIL.Image
from datasets import load_dataset
from datasets.utils.file_utils import get_datasets_user_agent
from torchvision.datasets import LSUN, CelebA

USER_AGENT = get_datasets_user_agent()


class ImageCaptionDataset:
    def __init__(self, configs, training=True):
        self.configs = configs
        self.training = training
        self.__resize_w = configs["resize_w"]
        self.__resize_h = configs["resize_h"]
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
            gcs_base_dir = "gs://celeb_a_dataset/"
            builder = tfds.builder("celeb_a", data_dir=gcs_base_dir, version='2.0.0')
            builder.download_and_prepare()
            if self.training:
                self.dataset = builder.as_dataset(split='train', as_supervised=False, shuffle_files=True, batch_size=1)
            else:
                self.dataset = builder.as_dataset(split='test', as_supervised=False, shuffle_files=False, batch_size=1)
        elif self.configs["name"] == "lsun_church":
            if self.training:
                self.dataset = LSUN(
                    path.abspath("data/lsun_church"), classes=["church_outdoor_train"]
                )
            else:
                self.dataset = LSUN(
                    path.abspath("data/lsun_church"), classes=["church_outdoor_val"]
                )
        else:
            raise ValueError(f'Invalid dataset name {self.configs["name"]}')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        caption = self.__get_caption(idx)
        image = self.__transform_img(self.dataset[idx][0])
        return {
            "image": image,
            "caption": caption,
        }

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
    
    def __transform_img(self, image):
        w, h = image.size 
        image = image.resize((self.__resize_w, self.__resize_h), resample=PIL.Image.LANCZOS)
        image = np.array(image).astype(np.float32) / 255.0
        image = image.transpose(2, 0, 1)
        image = torch.from_numpy(image)
        return 2.0 * image - 1.0
    
    def __get_caption(self, idx):
        if self.configs["name"] == "conceptual_captions":
            return self.dataset[idx]["caption"]
        elif self.configs["name"] == "celeba":
            return "A photo of a person"
        elif self.configs["name"] == "lsun_church":
            return "A photo of church exterior"
        else:
            raise ValueError(f'Invalid dataset name {self.configs["name"]}')


if __name__ == "__main__":
    configs = {
        "name": "lsun_church",
        "resize_w": 128,
        "resize_h": 128,
        }
    dataset = ImageCaptionDataset(configs, training=False)
    print(dataset[0])
