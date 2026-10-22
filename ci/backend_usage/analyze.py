#!/usr/bin/env python
#
# Summarize usage proxies per backend from the data `fetch.py` downloads, and
# print the tables as Markdown.
#
# None of these are direct measures of usage, so read them together:
#
# - Estimated human conda-forge downloads per month. Every `ibis-*` output,
#   including long-retired ones, gets a steady trickle of downloads from
#   mirrors and other automation, roughly the same for every version. That
#   floor is the median monthly download count of versions released at least
#   six months earlier; it's subtracted for every version-month before
#   averaging over the months the output existed. PyPI downloads can't be split
#   by extra, so backends whose users mostly install with pip (cloud
#   warehouses, in particular) are undercounted.
# - Public GitHub files outside this repo that call `ibis.<backend>.connect`.
#   This is a snapshot, not a time series, and misses private code entirely.
# - Issues (not pull requests) about a backend, matched by label or title,
#   and the distinct people who opened them. Backend labels stopped being
#   applied consistently in 2026, hence the title match.
# - Commits touching `ibis/backends/<backend>/`, and their distinct authors.
#   Cross-cutting refactors touch every backend, so this mostly measures
#   maintenance unless restricted to community contributors.
#
# "Community" excludes maintainers, Voltron Data staff (see `core.py`) and
# bots.
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from core import BACKENDS, CORE_EMAILS, CORE_LOGINS, TITLE_PATTERNS

import ibis
from ibis import _


def is_bot(col):
    return col.lower().re_search(r"\[bot\]|bot@|-bot\b|^renovate bot$")


def conda_per_month(con, data: Path, since):
    t = con.read_parquet(data / "conda_monthly.parquet")
    t = t.filter(t.data_source == "conda-forge", t.month >= since).mutate(
        backend=t.pkg_name.re_replace("^ibis-", "")
    )
    t = t.filter(t.backend.isin(BACKENDS))
    vm = t.group_by("backend", "pkg_version", "month").agg(n=_.counts.sum())
    release = vm.group_by("backend", "pkg_version").agg(release=_.month.min())
    vm = vm.join(release, ["backend", "pkg_version"])
    floor = (
        vm.filter(vm.month >= vm.release + ibis.interval(months=6))
        .group_by("backend")
        .agg(floor=_.n.median())
    )
    return (
        vm.group_by("backend")
        .agg(
            total=_.n.sum(),
            version_months=_.count(),
            months=_.month.nunique(),
            first=_.month.min(),
        )
        .join(floor, "backend")
        .select(
            "backend",
            conda_since=_.first.strftime("%Y-%m"),
            conda_per_month=((_.total - _.floor * _.version_months) / _.months).round(
                0
            ),
        )
    )


def backend_issues(con, data: Path, since):
    issues = con.read_json(data / "issues.jsonl")
    issues = issues.filter(
        ~issues.is_pr,
        issues.created_at.cast("timestamp") >= since,
        ~is_bot(issues.author),
        ~issues.author.isin(CORE_LOGINS),
    )
    patterns = ibis.memtable(
        {
            "backend": list(BACKENDS),
            "pattern": [TITLE_PATTERNS.get(b, b) for b in BACKENDS],
        }
    )
    return issues.cross_join(patterns).filter(
        _.labels.contains(_.backend) | _.title.lower().re_search(_.pattern)
    )


def backend_commits(con, data: Path, since):
    c = con.read_csv(data / "commits.csv")
    return c.filter(
        c.date.cast("date") >= since,
        c.backend.isin(BACKENDS),
        ~is_bot(c.email),
        ~is_bot(c.author),
        ~c.email.isin(CORE_EMAILS),
    )


def summary(con, data: Path, since):
    issues = (
        backend_issues(con, data, since)
        .group_by("backend")
        .agg(reporters=_.author.nunique(), issues=_.count())
    )
    per_author = (
        backend_commits(con, data, since)
        .group_by("backend", "email")
        .agg(n=_.sha.nunique())
    )
    commits = per_author.group_by("backend").agg(
        authors=_.count(), commits=_.n.sum(), repeat_authors=_.count(where=_.n >= 2)
    )
    search = con.read_csv(data / "code_search.csv")
    joined = ibis.memtable({"backend": list(BACKENDS)})
    for table in [conda_per_month(con, data, since), search, issues, commits]:
        joined = joined.left_join(table, "backend").drop("backend_right")
    df = joined.order_by(ibis.desc(_.conda_per_month.fill_null(-1))).to_pandas()
    counts = ["reporters", "issues", "authors", "commits", "repeat_authors"]
    df[counts] = df[counts].fillna(0)
    for col in ["conda_per_month", "repos_connect", *counts]:
        df[col] = df[col].astype("Int64")
    return df.rename(
        columns={
            "conda_per_month": "est. human conda dl/month",
            "conda_since": "conda data since",
            "repos_connect": "public files calling connect",
            "reporters": "community issue reporters",
            "issues": "community issues",
            "authors": "community commit authors",
            "commits": "community commits",
            "repeat_authors": "community authors with 2+ commits",
        }
    )


def reporters_by_year(con, data: Path, since):
    df = (
        backend_issues(con, data, since)
        .group_by("backend", year=_.created_at.cast("timestamp").year())
        .agg(n=_.author.nunique())
        .to_pandas()
        .pivot(index="backend", columns="year", values="n")
        .reindex(BACKENDS)
        .fillna(0)
        .astype(int)
    )
    return df.sort_values(list(df.columns), ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).parent / "data")
    parser.add_argument("--since", default="2023-01-01")
    args = parser.parse_args()
    since = ibis.date(args.since)
    con = ibis.duckdb.connect()

    print(f"### Usage proxies per backend, {args.since} to now\n")
    print(summary(con, args.data, since).to_markdown(index=False))
    print("\n### Community issue reporters per backend per year\n")
    print(reporters_by_year(con, args.data, since).to_markdown())


if __name__ == "__main__":
    pd.set_option("future.no_silent_downcasting", True)
    main()
