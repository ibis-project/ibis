#!/usr/bin/env python3
from __future__ import annotations

import concurrent.futures
import json
import os
import shutil
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping

import pooch
from google.cloud import storage

EXAMPLES_DIRECTORY = Path(__file__).parent


def make_registry(*, blobs: Iterable[storage.Blob], bucket: str, prefix: str) -> None:
    # Names of the data files
    path_prefix = prefix + "/"
    prefix_len = len(path_prefix)

    with tempfile.TemporaryDirectory() as directory:
        with concurrent.futures.ThreadPoolExecutor() as e:
            for fut in concurrent.futures.as_completed(
                e.submit(
                    pooch.retrieve,
                    url=f"https://storage.googleapis.com/{bucket}/{prefix}/{name[prefix_len:]}",
                    known_hash=None,
                    fname=name[prefix_len:],
                    path=directory,
                )
                for blob in blobs
                if (name := blob.name).startswith(path_prefix)
            ):
                fut.result()

        # Create the registry file from the downloaded data files
        pooch.make_registry(directory, EXAMPLES_DIRECTORY / "registry.txt")


def make_descriptions(descriptions_dir: Path) -> None:
    return (
        (file.name, file.read_text().strip()) for file in descriptions_dir.glob("*")
    )


def make_keys(registry: Path) -> dict:
    return (
        (key.split(os.extsep, maxsplit=1)[0], key)
        for key, _ in (
            row.split(maxsplit=1)
            for row in map(str.strip, registry.read_text().splitlines())
        )
    )


def make_metadata(
    *, descriptions: Path, registry: Path
) -> Mapping[str, dict[str, str]]:
    data = defaultdict(dict)

    for key, value in make_descriptions(descriptions):
        data[key]["description"] = value

    for key, value in make_keys(registry):
        data[key]["key"] = value

    return data


def add_wowah_example(client: storage.Client, *, data_path: Path):
    bucket = client.get_bucket("ibis-tutorial-data")

    args = []
    for blob in bucket.list_blobs(prefix="wowah_data"):
        name = blob.name
        if name.endswith("_raw.parquet"):
            tail = name.rsplit(os.sep, 1)[-1]
            path = data_path.joinpath(
                f"wowah_{tail}" if not tail.startswith("wowah") else tail
            )
            args.append((path, blob))

    with concurrent.futures.ThreadPoolExecutor() as e:
        for fut in concurrent.futures.as_completed(
            e.submit(
                lambda path, blob: path.write_bytes(blob.download_as_bytes()),
                path,
                blob,
            )
            for path, blob in args
        ):
            fut.result()


def main(args):
    bucket = args.bucket
    clean = args.clean

    data_path = EXAMPLES_DIRECTORY.joinpath("data")
    descriptions_path = EXAMPLES_DIRECTORY.joinpath("descriptions")

    if clean:
        shutil.rmtree(data_path, ignore_errors=True)
        shutil.rmtree(descriptions_path, ignore_errors=True)

    data_path.mkdir(parents=True, exist_ok=True)
    descriptions_path.mkdir(parents=True, exist_ok=True)

    client = storage.Client()

    add_wowah_example(client=client, data_path=data_path)

    # generate data from R
    subprocess.check_call(["Rscript", str(EXAMPLES_DIRECTORY / "gen_examples.R")])

    # rsync data and descriptions with the bucket
    subprocess.check_call(
        ["gsutil", "-m", "rsync", "-r", "-d", data_path, f"gs://{bucket}/data"]
    )

    # get bucket data and produce registry
    make_registry(blobs=client.list_blobs(bucket), bucket=bucket, prefix="data")

    data = make_metadata(
        descriptions=descriptions_path, registry=EXAMPLES_DIRECTORY / "registry.txt"
    )

    with EXAMPLES_DIRECTORY.joinpath("metadata.json").open(mode="w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="Set up the pooch registry from a GCS bucket.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "-b",
        "--bucket",
        default="ibis-examples",
        help="GCS bucket to rsync example data to",
    )
    p.add_argument(
        "-C",
        "--clean",
        action="store_true",
        help="Remove data and descriptions directories before generating examples",
    )

    main(p.parse_args())
