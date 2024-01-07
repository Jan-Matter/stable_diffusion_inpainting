# -*- coding: utf-8 -*-

from __future__ import division, print_function

import argparse
import shutil
import subprocess
from os import path, remove, rename
from pathlib import Path
from urllib.request import Request, urlopen
import os
from zipfile import ZipFile

__author__ = "Fisher Yu"
__email__ = "fy@cs.princeton.edu"
__license__ = "MIT"


def list_categories():
    url = "http://dl.yf.io/lsun/categories.txt"
    with urlopen(Request(url)) as response:
        return response.read().decode().strip().split("\n")


def download(out_dir, category, set_name, data_entry_limit):
    url = "http://dl.yf.io/lsun/scenes/{category}_" "{set_name}_lmdb.zip".format(
        **locals()
    )
    if set_name == "test":
        out_name = "test_lmdb.zip"
        url = "http://dl.yf.io/lsun/scenes/{set_name}_lmdb.zip"
    else:
        out_name = "{category}_{set_name}_lmdb.zip".format(**locals())
    out_path = path.join(out_dir, out_name)
    cmd = ["curl", "-C", "-", url, "-o", out_path]
    print("Downloading", category, set_name, "set")
    subprocess.call(cmd)

    if data_entry_limit is None:
        return

    cmd = ["unzip", out_path, "-d", path.dirname(out_path)]
    print("\nExtracting", category, set_name, "set")
    subprocess.call(cmd)

    remove(out_path)

    db_path = path.abspath(out_path)[0:-4]

    with lmdb.open(db_path, max_readers=1, map_size=int(3e9)) as env:
        txn = env.begin(write=True)
        db = txn.cursor()

        length = txn.stat()["entries"]

        if length <= data_entry_limit:
            return

        db = txn.cursor()

        # Iterate through the first n entries and delete them
        count = 0
        for _, _ in db:
            if count < (length - data_entry_limit):
                db.delete()
                count += 1
            else:
                break

        # Commit the transaction and close the environment
        txn.commit()
        db.close()

        print("\nCompacting", category, set_name, "set")

        compacted_path = f"{db_path}_compacted"

        Path(compacted_path).mkdir(parents=True, exist_ok=True)
        env.copy(path=compacted_path, compact=True)

    shutil.rmtree(db_path)
    rename(compacted_path, db_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--out_dir", default="")
    parser.add_argument("-c", "--category", default=None)
    parser.add_argument("-l", "--limit", type=int, default=40_000)
    args = parser.parse_args()

    categories = list_categories()
    if args.category is None:
        print("Downloading", len(categories), "categories")
        for category in categories:
            download(args.out_dir, category, "train")
            download(args.out_dir, category, "val")
        download(args.out_dir, "", "test")
    else:
        if args.category == "test":
            download(args.out_dir, "", "test", args.limit)
        elif args.category not in categories:
            print("Error:", args.category, "doesn't exist in", "LSUN release")
        else:
            download(args.out_dir, args.category, "train")
            download(args.out_dir, args.category, "val")
    
    for file in os.listdir("data/lsun_church"):
        if file.endswith(".zip"):
            zip_ref = ZipFile("data/lsun_church/" + file, 'r')
            zip_ref.extractall("data/lsun_church")
            zip_ref.close()
            os.remove("data/lsun_church/" + file)


if __name__ == "__main__":
    main()
