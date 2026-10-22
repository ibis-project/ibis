#!/usr/bin/env python
#
# Download the raw data behind `analyze.py`.
#
# - Monthly conda download counts for every `ibis*` package, from the public
#   dataset Anaconda publishes at
#   https://github.com/anaconda/anaconda-package-data.
# - Every issue and pull request in ibis-project/ibis, via the `gh` CLI.
# - The set of backend directories each commit touched, from the local git
#   history.
# - Counts of public GitHub files outside this repo that call
#   `ibis.<backend>.connect`, via the `gh` CLI. The code search API allows
#   about ten requests a minute, so this step takes a few minutes.
#
# Requires `ibis-framework[duckdb]` and an authenticated `gh` CLI.
from __future__ import annotations

import argparse
import csv
import datetime
import json
import re
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from core import BACKENDS

import ibis

CONDA_BASE = "https://anaconda-package-data.s3.amazonaws.com/conda/monthly"
REPO = "ibis-project/ibis"


def conda_urls(start_year: int) -> list[str]:
    urls = []
    today = datetime.date.today()
    for year in range(start_year, today.year + 1):
        for month in range(1, 13):
            url = f"{CONDA_BASE}/{year}/{year}-{month:02d}.parquet"
            request = urllib.request.Request(url, method="HEAD")
            try:
                urllib.request.urlopen(request)
            except urllib.error.HTTPError:
                return urls
            urls.append(url)
    return urls


def fetch_conda(out: Path, start_year: int) -> None:
    con = ibis.duckdb.connect()
    con.load_extension("httpfs")
    t = con.read_parquet(conda_urls(start_year), union_by_name=True)
    # `time` is a string, and older files only have the month (YYYY-MM).
    expr = (
        t.filter(t.pkg_name.startswith("ibis"))
        .mutate(month=(t.time.substr(0, 7) + "-01").cast("date"))
        .group_by("month", "pkg_name", "pkg_version", "data_source")
        .agg(counts=t.counts.sum())
    )
    con.to_parquet(expr, out / "conda_monthly.parquet")


def fetch_issues(out: Path) -> None:
    jq = (
        ".[] | {number, title, created_at, is_pr: (.pull_request != null), "
        "author: .user.login, labels: [.labels[].name]}"
    )
    with (out / "issues.jsonl").open("w") as f:
        subprocess.run(
            [
                "gh",
                "api",
                "--paginate",
                f"repos/{REPO}/issues?state=all&per_page=100",
                "--jq",
                jq,
            ],
            stdout=f,
            check=True,
        )


def fetch_commits(out: Path, rev: str, since: str) -> None:
    log = subprocess.run(
        [
            "git",
            "log",
            rev,
            f"--since={since}",
            "--no-merges",
            "--format=__C|%H|%as|%aN|%aE",
            "--name-only",
            "--",
            ":(top)ibis/backends/",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    rows = set()
    commit = None
    for line in log.splitlines():
        if line.startswith("__C|"):
            commit = tuple(line.split("|", 4)[1:])
        elif commit and (m := re.match(r"ibis/backends/([^/]+)/", line)):
            rows.add((*commit, m.group(1)))
    with (out / "commits.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sha", "date", "author", "email", "backend"])
        writer.writerows(sorted(rows))


def fetch_code_search(out: Path) -> None:
    with (out / "code_search.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["backend", "repos_connect"])
        for backend in BACKENDS:
            query = f'"ibis.{backend}.connect" language:Python -repo:{REPO}'
            result = subprocess.run(
                ["gh", "api", "-X", "GET", "search/code", "-f", f"q={query}"],
                capture_output=True,
                text=True,
                check=True,
            )
            writer.writerow([backend, json.loads(result.stdout)["total_count"]])
            f.flush()
            time.sleep(7)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).parent / "data")
    parser.add_argument(
        "--rev", default="HEAD", help="git revision to read history from"
    )
    parser.add_argument("--start-year", type=int, default=2022)
    parser.add_argument(
        "--skip",
        nargs="*",
        default=[],
        choices=["conda", "issues", "commits", "code-search"],
    )
    args = parser.parse_args()
    args.data.mkdir(parents=True, exist_ok=True)

    steps = {
        "conda": lambda: fetch_conda(args.data, args.start_year),
        "issues": lambda: fetch_issues(args.data),
        "commits": lambda: fetch_commits(
            args.data, args.rev, f"{args.start_year}-01-01"
        ),
        "code-search": lambda: fetch_code_search(args.data),
    }
    for name, step in steps.items():
        if name not in args.skip:
            print(f"fetching {name}...", flush=True)
            step()


if __name__ == "__main__":
    main()
