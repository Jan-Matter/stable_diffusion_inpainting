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


class CustomImageCaptionDataset:
    def __init__(self, configs, training=True):
        self.configs = configs
        self.training = training
        self.__resize_w = configs["resize_w"]
        self.__resize_h = configs["resize_h"]
        

    def __len__(self):
        if self.configs["name"] == "maskara":
            return len(list(Path("data/custom_datasets/maskara_3").glob("*.jpg")))

    def __getitem__(self, idx):
        caption = self.__get_caption(idx)
        image = self.__get_image(idx)
        return {
            "image": image,
            "caption": caption,
        }
    
    def __transform_img(self, image):
        w, h = image.size 
        image = image.resize((self.__resize_w, self.__resize_h), resample=PIL.Image.LANCZOS)
        image = np.array(image).astype(np.float32) / 255.0
        image = image.transpose(2, 0, 1)
        image = torch.from_numpy(image)
        return 2.0 * image - 1.0
    
    
    def __get_caption(self, idx):
        if self.configs["name"] == "maskara":
            return "A photo of a man"
    
    def __get_image(self, idx):
        if self.configs["name"] == "maskara":
            img_folder_path = "data/custom_datasets/maskara_3"
            img_paths = list(Path(img_folder_path).glob("*.jpg"))
            img_path = img_paths[idx]
            image = PIL.Image.open(img_path)
            transformed_image = self.__transform_img(image)
            return transformed_image
        else:
            raise NotImplementedError("Dataset not implemented yet.")



if __name__ == "__main__":
    configs = {
        "name": "maskara",
        "resize_w": 128,
        "resize_h": 128,
        }
    dataset = ImageCaptionDataset(configs, training=False)
    output = dataset[0]
    print(output)