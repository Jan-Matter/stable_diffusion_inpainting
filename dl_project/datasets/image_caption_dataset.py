"""Module containing the Dataset for image captioning tasks."""
import io
import multiprocessing
import urllib
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import PIL.Image
from datasets import load_dataset
from datasets.utils.file_utils import get_datasets_user_agent
from torchvision.datasets import LSUN, CelebA

USER_AGENT = get_datasets_user_agent()


def fetch_single_image(image_url, timeout=None, retries=0):
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


def fetch_images(batch, num_threads, timeout=None, retries=0):
    fetch_single_image_with_args = partial(
        fetch_single_image, timeout=timeout, retries=retries
    )

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        batch["image"] = list(
            executor.map(fetch_single_image_with_args, batch["image_url"])
        )

    return batch


conceptual_dataset = load_dataset("conceptual_captions")
conceptual_dataset = conceptual_dataset.map(
    fetch_images,
    batched=True,
    batch_size=100,
    fn_kwargs={"num_threads": multiprocessing.cpu_count()},
)

conceptual_dataset.with_format("torch")

celeba_dataset = CelebA("data/celeba")
lsun_church_dataset = LSUN("data/lsun_church", "church")
