#!/usr/bin/env python3
from __future__ import annotations

import concurrent.futures
import io
import json
import os
import shutil
import subprocess
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping

import pooch
import requests
from google.cloud import storage

import ibis

EXAMPLES_DIRECTORY = Path(__file__).parent


def make_registry(*, blobs: Iterable[storage.Blob], bucket: str, prefix: str) -> None:
    # Names of the data files
    path_prefix = f"{prefix}/"
    prefix_len = len(path_prefix)

    base_url = "https://storage.googleapis.com"

    with tempfile.TemporaryDirectory() as directory:
        with concurrent.futures.ThreadPoolExecutor() as e:
            for fut in concurrent.futures.as_completed(
                e.submit(
                    pooch.retrieve,
                    url=f"{base_url}/{bucket}/{prefix}/{name[prefix_len:]}",
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


def add_movielens_example(data_path: Path):
    filename = "ml-latest-small.zip"
    resp = requests.get(f"https://files.grouplens.org/datasets/movielens/{filename}")
    resp.raise_for_status()
    raw_bytes = resp.content

    # convert to parquet
    with tempfile.TemporaryDirectory() as d:
        con = ibis.duckdb.connect(Path(d, "movielens.ddb"), experimental_parallel_csv=1)
        d = Path(d)
        all_data = d / filename
        all_data.write_bytes(raw_bytes)

        # extract the CSVs into the current temp dir and convert them to
        # zstd-compressed Parquet files using DuckDB
        with zipfile.ZipFile(all_data) as zf:
            members = [name for name in zf.namelist() if name.endswith(".csv")]
            zf.extractall(d, members=members)

        for member, csv_path in zip(members, map(d.joinpath, members)):
            parquet_path = data_path.joinpath(
                member.replace("ml-latest-small/", "ml_latest_small_")
            ).with_suffix(".parquet")
            con.read_csv(csv_path).to_parquet(parquet_path, codec="zstd")


def add_imdb_example(*, source_dir: Path | None, data_path: Path) -> None:
    def convert_to_parquet(con, tsv_file: Path, description: str) -> None:
        dest = data_path.joinpath(
            "imdb_"
            + tsv_file.with_suffix("").with_suffix(".parquet").name.replace(".", "_", 1)
        )
        con.read_csv(tsv_file, nullstr=r"\N", header=1, quote="").to_parquet(
            dest, compression="zstd"
        )
        dest.parent.parent.joinpath(
            "descriptions", dest.with_suffix("").name
        ).write_text(description)
        print(f"converted {tsv_file.name} to parquet")  # noqa: T201

    def download_file(base: str, outdir: Path) -> None:
        resp = requests.get(f"https://datasets.imdbws.com/{base}", stream=True)
        resp.raise_for_status()

        with outdir.joinpath(base).open(mode="wb") as f:
            for chunk in resp.iter_content(chunk_size=io.DEFAULT_BUFFER_SIZE):
                f.write(chunk)
        print(f"downloaded {base}")  # noqa: T201

    meta = {
        "name.basics.tsv.gz": """\
Contains the following information for names:
* nconst (string) - alphanumeric unique identifier of the name/person
* primaryName (string) - name by which the person is most often credited
* birthYear - in YYYY format
* deathYear - in YYYY format if applicable, else '\\N'
* primaryProfession (array of strings) - the top-3 professions of the person
* knownForTitles (array of tconsts) - titles the person is known for""",
        "title.akas.tsv.gz": """\
Contains the following information for titles:
* titleId (string) - a tconst, an alphanumeric unique identifier of the title
* ordering (integer) - a number to uniquely identify rows for a given titleId
* title (string) - the localized title
* region (string) - the region for this version of the title
* language (string) - the language of the title
* types (array) - Enumerated set of attributes for this alternative title. One or more of the following: "alternative", "dvd", "festival", "tv", "video", "working", "original", "imdbDisplay". New values may be added in the future without warning
* attributes (array) - Additional terms to describe this alternative title, not enumerated
* isOriginalTitle (boolean) - 0: not original title; 1: original title""",
        "title.basics.tsv.gz": """\
Contains the following information for titles:
* tconst (string) - alphanumeric unique identifier of the title
* titleType (string) - the type/format of the title (e.g. movie, short, tvseries, tvepisode, video, etc)
* primaryTitle (string) - the more popular title / the title used by the filmmakers on promotional materials at the point of release
* originalTitle (string) - original title, in the original language
* isAdult (boolean) - 0: non-adult title; 1: adult title
* startYear (YYYY) - represents the release year of a title. In the case of TV Series, it is the series start year
* endYear (YYYY) - TV Series end year. '\\N' for all other title types
* runtimeMinutes - primary runtime of the title, in minutes
* genres (string array) - includes up to three genres associated with the title""",
        "title.crew.tsv.gz": """\
Contains the director and writer information for all the titles in IMDb. Fields include:
* tconst (string) - alphanumeric unique identifier of the title
* directors (array of nconsts) - director(s) of the given title
* writers (array of nconsts) - writer(s) of the given title""",
        "title.episode.tsv.gz": """\
Contains the tv episode information. Fields include:
* tconst (string) - alphanumeric identifier of episode
* parentTconst (string) - alphanumeric identifier of the parent TV Series
* seasonNumber (integer) - season number the episode belongs to
* episodeNumber (integer) - episode number of the tconst in the TV series""",
        "title.principals.tsv.gz": """\
Contains the principal cast/crew for titles
* tconst (string) - alphanumeric unique identifier of the title
* ordering (integer) - a number to uniquely identify rows for a given titleId
* nconst (string) - alphanumeric unique identifier of the name/person
* category (string) - the category of job that person was in
* job (string) - the specific job title if applicable, else '\\N'
* characters (string) - the name of the character played if applicable, else '\\N'""",
        "title.ratings.tsv.gz": """\
Contains the IMDb rating and votes information for titles
* tconst (string) - alphanumeric unique identifier of the title
* averageRating - weighted average of all the individual user ratings
* numVotes - number of votes the title has received""",
    }

    with tempfile.TemporaryDirectory() as d:
        if source_dir is None:
            source_dir = Path(d)

            with concurrent.futures.ThreadPoolExecutor() as e:
                for fut in concurrent.futures.as_completed(
                    e.submit(download_file, base=base, outdir=source_dir)
                    for base in meta.keys()
                ):
                    fut.result()

        con = ibis.duckdb.connect(source_dir / "imdb.ddb", experimental_parallel_csv=1)

        with concurrent.futures.ThreadPoolExecutor() as e:
            for fut in concurrent.futures.as_completed(
                e.submit(convert_to_parquet, con, path, description=meta[path.name])
                for path in source_dir.glob("*.tsv.gz")
            ):
                fut.result()


def main(args):
    bucket = args.bucket
    clean = args.clean

    data_path = EXAMPLES_DIRECTORY / "data"
    descriptions_path = EXAMPLES_DIRECTORY / "descriptions"

    if clean:
        shutil.rmtree(data_path, ignore_errors=True)
        shutil.rmtree(descriptions_path, ignore_errors=True)

    data_path.mkdir(parents=True, exist_ok=True)
    descriptions_path.mkdir(parents=True, exist_ok=True)

    add_movielens_example(data_path)

    add_imdb_example(
        source_dir=(
            source_dir
            if (source_dir := args.imdb_source_dir) is None
            else Path(source_dir)
        ),
        data_path=data_path,
    )

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
    p.add_argument(
        "-I",
        "--imdb-source-dir",
        help="Directory containing imdb source data",
        default=None,
        type=str,
    )

    main(p.parse_args())
