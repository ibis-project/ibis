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
#   usage: ./apache-pull_request_number-merge.py    (see config env vars below)
#
# Lightly modified from version of this script in incubator-parquet-format

"""Command line tool for merging PRs."""

import collections
import pathlib
import textwrap

import click

import plumbum

from plumbum import cmd

import requests

IBIS_HOME = pathlib.Path(__file__).parent.parent
GITHUB_API_BASE = "https://api.github.com/repos/ibis-project/ibis"

git = cmd.git["-C", IBIS_HOME]


def merge_pr(
    pr_num: int,
    base_ref: str,
    target_ref: str,
    commit_title: str,
    body: str,
    pr_repo_desc: str,
    original_head: str,
    remote: str,
    merge_method: str,
    github_user: str,
    password: str,
) -> None:
    """Merge a pull request."""
    git_log = git["log", f"{remote}/{target_ref}..{base_ref}"]

    commit_authors = git_log["--pretty=format:%an <%ae>"]().splitlines()
    author_count = collections.Counter(commit_authors)
    distinct_authors = [author for author, _ in author_count.most_common()]
    commits = git_log["--pretty=format:%h [%an] %s"]().splitlines()

    merge_message_pieces = []
    if body:
        merge_message_pieces.append("\n".join(textwrap.wrap(body)))
    merge_message_pieces.extend(map("Author: {}".format, distinct_authors))

    # The string f"Closes #{pull_request_number:d}" is required for GitHub to
    # correctly close the PR
    merge_message_pieces.append(
        f"\nCloses #{pr_num:d} from {pr_repo_desc} and squashes the following "
        "commits:\n"
    )
    merge_message_pieces += commits

    commit_message = (
        "\n".join(merge_message_pieces)
        .encode("unicode_escape")
        .decode("UTF-8")
    )
    # PUT /repos/:owner/:repo/pulls/:number/merge
    resp = requests.put(
        f"{GITHUB_API_BASE}/pulls/{pr_num:d}/merge",
        data=dict(
            commit_title=commit_title,
            commit_message=commit_message,
            merge_method=merge_method,
        ),
        auth=(github_user, password),
    )
    resp.raise_for_status()
    if resp.status_code == 200:
        resp_json = resp.json()
        merged = resp_json["merged"]
        assert merged is True, merged
        click.echo(f"Pull request #{pr_num:d} successfully merged.")


@click.command()
@click.option(
    "-p",
    "--pull-request-number",
    type=int,
    prompt="Which pull request would you like to merge? (e.g., 34)",
    help="The pull request number to merge.",
)
@click.option(
    "-M",
    "--merge-method",
    type=click.Choice(("merge", "squash", "rebase")),
    default="squash",
    help="The method to use for merging the PR.",
    show_default=True,
)
@click.option(
    "-r",
    "--remote",
    default="upstream",
    help="A valid git remote.",
    show_default=True,
)
@click.option("-u", "--github-user", help="Your GitHub user name.")
@click.option(
    "-P",
    "--password",
    help="Your GitHub password for authentication and authorization.",
)
def main(
    pull_request_number: int,
    merge_method: str,
    remote: str,
    github_user: str,
    password: str,
) -> None:  # noqa: D103
    try:
        git["fetch", remote]()
    except plumbum.commands.processes.ProcessExecutionError as e:
        raise click.ClickException(e.stderr)

    original_head = git["rev-parse", "--abbrev-ref", "HEAD"]().strip()

    if not original_head:
        original_head = git["rev-parse", "HEAD"]().strip()

    resp = requests.get(f"{GITHUB_API_BASE}/pulls/{pull_request_number:d}")
    resp.raise_for_status()
    pr_json = resp.json()

    message = pr_json.get("message", None)
    if message is not None and message.lower() == "not found":
        raise click.ClickException(
            f"PR {pull_request_number:d} does not exist."
        )

    if not pr_json["mergeable"]:
        raise click.ClickException(
            "Pull request {:d} cannot be merged in its current form."
        )

    url = pr_json["url"]
    commit_title = pr_json["title"]
    body = pr_json["body"]
    target_ref = pr_json["base"]["ref"]
    user_login = pr_json["user"]["login"]
    base_ref = pr_json["head"]["ref"]
    pr_repo_desc = f"{user_login}/{base_ref}"

    click.echo(f"=== Pull Request #{pull_request_number:d} ===")
    click.echo(
        f"title\t{commit_title}\n"
        f"source\t{pr_repo_desc}\n"
        f"target\t{remote}/{target_ref}\n"
        f"url\t{url}"
    )

    merge_pr(
        pull_request_number,
        base_ref,
        target_ref,
        commit_title,
        body,
        pr_repo_desc,
        original_head,
        remote,
        merge_method,
        github_user,
        password,
    )


if __name__ == "__main__":
    main()
