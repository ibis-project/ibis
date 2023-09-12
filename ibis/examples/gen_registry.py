#!/usr/bin/env python3
from __future__ import annotations

import concurrent.futures
import functools
import json
import os
import subprocess
import tempfile
import zipfile
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pins
import requests
import tqdm
from google.cloud import storage

import ibis

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

EXAMPLES_DIRECTORY = Path(__file__).parent

pins.config.pins_options.quiet = True


Metadata = dict[str, dict[str, str] | None]


def make_descriptions(descriptions_dir: Path) -> Iterator[tuple[str, str]]:
    return (
        (file.name, file.read_text().strip()) for file in descriptions_dir.glob("*")
    )


def make_keys(registry: Path) -> dict[str, str]:
    return (
        (key.split(os.extsep, maxsplit=1)[0], key)
        for key, _ in (
            row.split(maxsplit=1)
            for row in map(str.strip, registry.read_text().splitlines())
        )
    )


def add_wowah_example(data_path, *, client: storage.Client, metadata: Metadata) -> None:
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
            metadata[path.with_suffix("").name] = {}

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


def add_movielens_example(
    data_path: Path, *, metadata: Metadata, source_zip: Path | None = None
):
    filename = "ml-latest-small.zip"

    if source_zip is not None and source_zip.exists():
        raw_bytes = source_zip.read_bytes()
    else:
        resp = requests.get(
            f"https://files.grouplens.org/datasets/movielens/{filename}"
        )
        resp.raise_for_status()
        raw_bytes = resp.content

    # convert to parquet
    with tempfile.TemporaryDirectory() as d:
        con = ibis.duckdb.connect()
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
            metadata[parquet_path.with_suffix("").name] = {}
            con.read_csv(csv_path).to_parquet(parquet_path, codec="zstd")


def add_imdb_example(data_path: Path) -> None:
    def convert_to_parquet(
        base: Path,
        *,
        con: ibis.backends.duckdb.Base,
        description: str,
        bar: tqdm.tqdm,
    ) -> None:
        dest = data_path.joinpath(
            "imdb_"
            + Path(base)
            .with_suffix("")
            .with_suffix(".parquet")
            .name.replace(".", "_", 1)
        )
        con.read_csv(
            f"https://datasets.imdbws.com/{base}",
            nullstr=r"\N",
            header=1,
            quote="",
        ).to_parquet(dest, compression="zstd")
        dest.parents[1].joinpath("descriptions", dest.with_suffix("").name).write_text(
            description
        )
        bar.update()

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

    bar = tqdm.tqdm(total=len(meta))
    with concurrent.futures.ThreadPoolExecutor() as e:
        for fut in concurrent.futures.as_completed(
            e.submit(
                convert_to_parquet,
                base,
                con=ibis.duckdb.connect(),
                description=description,
                bar=bar,
            )
            for base, description in meta.items()
        ):
            fut.result()


def main(parser):
    args = parser.parse_args()

    data_path = EXAMPLES_DIRECTORY / "data"
    descriptions_path = EXAMPLES_DIRECTORY / "descriptions"

    data_path.mkdir(parents=True, exist_ok=True)
    descriptions_path.mkdir(parents=True, exist_ok=True)

    metadata = {}

    add_movielens_example(
        data_path,
        metadata=metadata,
        source_zip=(
            Path(ml_source_zip)
            if (ml_source_zip := args.movielens_source_zip) is not None
            else None
        ),
    )

    add_imdb_example(data_path)

    add_wowah_example(data_path, client=storage.Client(), metadata=metadata)

    # generate data from R
    subprocess.check_call(["Rscript", str(EXAMPLES_DIRECTORY / "gen_examples.R")])

    verify_case(parser, metadata)

    if not args.dry_run:
        board = pins.board_gcs(args.bucket)

        def write_pin(
            path: Path,
            *,
            board: pins.Board,
            metadata: Metadata,
            bar: tqdm.tqdm,
        ) -> None:
            pathname = path.name
            suffixes = path.suffixes
            name = pathname[: -sum(map(len, suffixes))]
            description = metadata.get(name, {}).get("description")
            board.pin_upload(
                paths=[str(path)],
                name=name,
                title=f"`{pathname}` dataset",
                description=description,
            )
            bar.update()

        data_paths = list(data_path.glob("*"))

        write_pin = functools.partial(
            write_pin,
            board=board,
            metadata=metadata,
            bar=tqdm.tqdm(total=len(data_paths)),
        )

        with concurrent.futures.ThreadPoolExecutor() as e:
            for fut in concurrent.futures.as_completed(
                e.submit(write_pin, path) for path in data_paths
            ):
                fut.result()

        metadata.update(
            (key, {"description": value})
            for key, value in make_descriptions(descriptions_path)
        )

        with EXAMPLES_DIRECTORY.joinpath("metadata.json").open(mode="w") as f:
            json.dump(metadata, f, indent=2, sort_keys=True)
            f.write("\n")


def verify_case(parser: argparse.ArgumentParser, data: Mapping[str, Any]) -> None:
    counter = Counter(map(str.lower, data.keys()))

    invalid_keys = [key for key, count in counter.items() if count > 1]
    if invalid_keys:
        parser.error(
            f"keys {invalid_keys} are incompatible with case-insensitive file systems"
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Set up the pin board from a GCS bucket.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-b",
        "--bucket",
        default=ibis.examples._BUCKET,
        help="GCS bucket in which to store data",
    )
    parser.add_argument(
        "-I",
        "--imdb-source-dir",
        help="Directory containing imdb source data",
        default=None,
        type=str,
    )
    parser.add_argument(
        "-M",
        "--movielens-source-zip",
        help="MovieLens data zip file",
        default=None,
        type=str,
    )
    parser.add_argument(
        "-d",
        "--dry-run",
        action="store_true",
        help="Avoid executing any code that writes to the example data bucket",
    )

    main(parser)
