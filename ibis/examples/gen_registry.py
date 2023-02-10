#!/usr/bin/env python3
import concurrent.futures
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable

import pooch
from google.cloud import storage

EXAMPLES_DIRECTORY = Path(__file__).parent


def make_registry(*, blobs: Iterable[storage.Blob], bucket: str, prefix: str) -> None:
    # Names of the data files
    path_prefix = prefix + "/"
    prefix_len = len(path_prefix)
    filenames = [
        name[prefix_len:]
        for blob in blobs
        if (name := blob.name).startswith(path_prefix)
    ]

    with tempfile.TemporaryDirectory() as directory:
        with concurrent.futures.ThreadPoolExecutor() as e:
            for fut in concurrent.futures.as_completed(
                e.submit(
                    pooch.retrieve,
                    url=f"https://storage.googleapis.com/{bucket}/{prefix}/{fname}",
                    known_hash=None,
                    fname=fname,
                    path=directory,
                )
                for fname in filenames
            ):
                fut.result()

        # Create the registry file from the downloaded data files
        pooch.make_registry(directory, EXAMPLES_DIRECTORY / "registry.txt")


def make_descriptions(descriptions_directory: Path):
    data = {
        file.name: file.read_text().strip()
        for file in sorted(descriptions_directory.glob("*"))
    }
    json_path = EXAMPLES_DIRECTORY / "descriptions.json"
    with json_path.open(mode="w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    return json_path


def main(args):
    bucket = args.bucket

    # generate data from R
    data_path = EXAMPLES_DIRECTORY.joinpath("data")
    data_path.mkdir(parents=True, exist_ok=True)

    descriptions_path = EXAMPLES_DIRECTORY.joinpath("descriptions")
    descriptions_path.mkdir(parents=True, exist_ok=True)

    subprocess.check_call(["Rscript", str(EXAMPLES_DIRECTORY / "gen_examples.R")])
    descriptions_json = make_descriptions(descriptions_path)

    # rsync data and descriptions with the bucket
    subprocess.check_call(["gsutil", "-m", "rsync", data_path, f"gs://{bucket}/data"])
    subprocess.check_call(
        ["gsutil", "cp", str(descriptions_json), f"gs://{bucket}/descriptions.json"]
    )

    # get bucket data and produce registry
    client = storage.Client()
    blobs = list(client.list_blobs(bucket))
    make_registry(blobs=blobs, bucket=bucket, prefix="data")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="Set up the pooch registry from a GCS bucket.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-b", "--bucket", default="ibis-examples", help="GCS bucket")

    main(p.parse_args())
