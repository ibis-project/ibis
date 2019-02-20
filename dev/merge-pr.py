#!/usr/bin/env python
#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# Utility for creating well-formed pull request merges and pushing them to
# Apache.
#   usage: ./apache-pr-merge.py    (see config env vars below)
#
# Lightly modified from version of this script in incubator-parquet-format

"""Command line tool for merging PRs."""

import collections
import contextlib
import os
import pathlib
import sys
import textwrap

from typing import Iterator

import click

from plumbum import cmd

import requests

IBIS_HOME = pathlib.Path(__file__).parent.parent
PROJECT_NAME = "ibis"

# Remote name with the PR
PR_REMOTE_NAME = os.environ.get("PR_REMOTE_NAME", "upstream")

# Remote name where results pushed
PUSH_REMOTE_NAME = os.environ.get("PUSH_REMOTE_NAME", "upstream")

GITHUB_API_BASE = "https://api.github.com/repos/ibis-project/{}".format(
    PROJECT_NAME
)

# Prefix added to temporary branches
BRANCH_PREFIX = "PR_TOOL"

git = cmd.git["-C", IBIS_HOME]


@contextlib.contextmanager
def clean_up(new_head: str, original_head: str) -> Iterator[None]:
    """Checkout `new_head` and return to `original_head` after yielding."""
    git["checkout", new_head](stdout=sys.stdout, stderr=sys.stderr)
    try:
        yield
    finally:
        git["checkout", original_head](stdout=sys.stdout, stderr=sys.stderr)
        branches = git["branch"]().strip().split()
        for branch in branches:
            if branch.startswith(BRANCH_PREFIX):
                git["branch", "-D", branch](
                    stdout=sys.stdout, stderr=sys.stderr
                )


def merge_pr(
    pr_num: int,
    target_ref: str,
    title: str,
    body: str,
    pr_repo_desc: str,
    original_head: str,
    target_branch_name: str,
    pr_branch_name: str,
    confirm_push: bool,
) -> None:
    """Merge a pull request."""
    git["merge", pr_branch_name, "--squash"](
        stdout=sys.stdout, stderr=sys.stderr
    )

    commit_authors = git[
        "log", "HEAD..{}".format(pr_branch_name), "--pretty=format:%an <%ae>"
    ]().split()
    author_count = collections.Counter(commit_authors)
    distinct_authors = [author for author, _ in author_count.most_common()]
    primary_author = distinct_authors[0]
    commits = git[
        "log", "HEAD..{}".format(pr_branch_name), "--pretty=format:%h [%an] %s"
    ]().split()

    merge_message_flags = ["-m", title]
    if body is not None:
        merge_message_flags += ["-m", "\n".join(textwrap.wrap(body))]

    authors = "\n".join(map("Author: {}".format, distinct_authors))

    merge_message_flags += ["-m", authors]

    # The string "Closes #{pr}" string is required for GitHub to correctly
    # close the PR
    merge_message_flags += [
        "-m",
        "Closes #{:d} from {} and squashes the following commits:".format(
            pr_num, pr_repo_desc
        ),
    ]
    for commit in commits:
        merge_message_flags += ["-m", commit]

    git["commit", "--no-verify", "--author", primary_author][
        merge_message_flags
    ](stdout=sys.stdout, stderr=sys.stderr)

    if confirm_push:
        prompt = "Merge complete (local ref {}). Push to {}?".format(
            target_branch_name, PUSH_REMOTE_NAME
        )
        if input("\n{} ([Yy]/n): ".format(prompt)).lower() != "y":
            sys.exit(-1)

    git[
        "push",
        PUSH_REMOTE_NAME,
        "{}:{}".format(target_branch_name, target_ref),
    ](stdout=sys.stdout, stderr=sys.stderr)

    merge_hash = git["rev-parse", target_branch_name]().strip()
    click.echo("Pull request #{:d} merged!".format(pr_num))
    click.echo("Merge hash: {}".format(merge_hash))


@click.command()
@click.option(
    "-p",
    "--pr",
    type=int,
    prompt="Which pull request would you like to merge? (e.g. 34)",
)
@click.option("--confirm-push/--no-confirm-push", default=True)
def main(pr: int, confirm_push: bool) -> None:  # noqa: D103
    original_head = git["rev-parse", "--abbrev-ref", "HEAD"]().strip()

    if not original_head:
        original_head = git["rev-parse", "HEAD"]().strip()

    pr_json = requests.get("{}/pulls/{:d}".format(GITHUB_API_BASE, pr)).json()

    message = pr_json.get("message", None)
    if message is not None and message.lower() == "not found":
        raise click.ClickException("PR {:d} does not exist.".format(pr))

    if not pr_json["mergeable"]:
        raise click.ClickException(
            "Pull request {:d} cannot be merged in its current form."
        )

    url = pr_json["url"]
    title = pr_json["title"]
    body = pr_json["body"]
    target_ref = pr_json["base"]["ref"]
    user_login = pr_json["user"]["login"]
    base_ref = pr_json["head"]["ref"]
    pr_repo_desc = "{}/{}".format(user_login, base_ref)

    click.echo("\n=== Pull Request #{:d} ===".format(pr))
    click.echo(
        "title\t{}\nsource\t{}\ntarget\t{}\nurl\t{}".format(
            title, pr_repo_desc, target_ref, url
        )
    )

    pr_branch_name = "{}_MERGE_PR_{:d}".format(BRANCH_PREFIX, pr)
    target_branch_name = "{}_MERGE_PR_{:d}_{}".format(
        BRANCH_PREFIX, pr, target_ref.upper()
    )
    git[
        "fetch", PR_REMOTE_NAME, "pull/{:d}/head:{}".format(pr, pr_branch_name)
    ](stdout=sys.stdout, stderr=sys.stderr)
    git[
        "fetch",
        PUSH_REMOTE_NAME,
        "{}:{}".format(target_ref, target_branch_name),
    ](stdout=sys.stdout, stderr=sys.stderr)
    with clean_up(target_branch_name, original_head):
        merge_pr(
            pr,
            target_ref,
            title,
            body,
            pr_repo_desc,
            original_head,
            target_branch_name,
            pr_branch_name,
            confirm_push,
        )


if __name__ == "__main__":
    main()
