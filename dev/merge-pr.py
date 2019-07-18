#!/usr/bin/env python

"""Command line tool for merging PRs."""

import collections
import pathlib
import sys
import textwrap

import click
import plumbum
import requests
from plumbum import cmd

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
    git_log = git[
        "log",
        "{remote}/{target_ref}..{base_ref}".format(
            remote=remote, target_ref=target_ref, base_ref=base_ref
        ),
    ]

    commit_authors = git_log["--pretty=format:%an <%ae>"]().splitlines()
    author_count = collections.Counter(commit_authors)
    distinct_authors = [author for author, _ in author_count.most_common()]
    commits = git_log["--pretty=format:%h [%an] %s"]().splitlines()

    merge_message_pieces = []
    if body:
        merge_message_pieces.append("\n".join(textwrap.wrap(body)))
    merge_message_pieces.extend(map("Author: {}".format, distinct_authors))

    # The string "Closes #{pull_request_number:d}" is required for GitHub to
    # correctly close the PR
    merge_message_pieces.append(
        (
            "\nCloses #{pr_num:d} from {pr_repo_desc} and squashes the "
            "following commits:\n"
        ).format(pr_num=pr_num, pr_repo_desc=pr_repo_desc)
    )
    merge_message_pieces += commits

    commit_message = "\n".join(merge_message_pieces)
    resp = requests.put(
        "{GITHUB_API_BASE}/pulls/{pr_num:d}/merge".format(
            GITHUB_API_BASE=GITHUB_API_BASE, pr_num=pr_num
        ),
        json=dict(
            commit_title=commit_title,
            commit_message=commit_message,
            merge_method=merge_method,
        ),
        auth=(github_user, password),
    )
    status_code = resp.status_code
    if status_code == 200:
        resp_json = resp.json()
        assert resp_json["merged"]
        click.echo(resp_json["message"])
    elif status_code == 405 or status_code == 409:
        resp_json = resp.json()
        raise click.ClickException(resp_json["message"])
    else:
        resp.raise_for_status()


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
    try:
        git[
            "fetch",
            remote,
            "pull/{pull_request_number:d}/head".format(
                pull_request_number=pull_request_number
            ),
        ]()
    except plumbum.commands.processes.ProcessExecutionError as e:
        raise click.ClickException(e.stderr)

    original_head = git["rev-parse", "--abbrev-ref", "HEAD"]().strip()

    if not original_head:
        original_head = git["rev-parse", "HEAD"]().strip()

    resp = requests.get(
        "{GITHUB_API_BASE}/pulls/{pull_request_number:d}".format(
            GITHUB_API_BASE=GITHUB_API_BASE,
            pull_request_number=pull_request_number,
        )
    )
    if resp.status_code == 404:
        pr_json = resp.json()
        message = pr_json.get("message", None)
        if message is not None:
            raise click.ClickException(
                "PR {pull_request_number:d} does not exist.".format(
                    pull_request_number=pull_request_number
                )
            )
    else:
        resp.raise_for_status()

    pr_json = resp.json()

    # no-op if already merged
    if pr_json["merged"]:
        click.echo(
            "#{pr_num:d} already merged. Nothing to do.".format(
                pr_num=pull_request_number
            )
        )
        sys.exit(0)

    if not pr_json["mergeable"]:
        raise click.ClickException(
            (
                "Pull request #{pr_num:d} cannot be merged in its current "
                "form. See "
                "https://github.com/ibis-project/ibis/pulls/{pr_num:d} for "
                "more details."
            ).format(pr_num=pull_request_number)
        )

    url = pr_json["url"]
    commit_title = pr_json["title"]
    body = pr_json["body"]
    target_ref = pr_json["base"]["ref"]
    user_login = pr_json["user"]["login"]
    base_ref = pr_json["head"]["ref"]
    pr_repo_desc = "{user_login}/{base_ref}".format(
        user_login=user_login, base_ref=base_ref
    )

    click.echo(
        "=== Pull Request #{pull_request_number:d} ===".format(
            pull_request_number=pull_request_number
        )
    )
    click.echo(
        (
            "title\t{commit_title}\n"
            "source\t{pr_repo_desc}\n"
            "target\t{remote}/{target_ref}\n"
            "url\t{url}"
        ).format(
            commit_title=commit_title,
            pr_repo_desc=pr_repo_desc,
            remote=remote,
            target_ref=target_ref,
            url=url,
        )
    )

    base_ref_commit = (
        git[
            "ls-remote",
            remote,
            "refs/pull/{pull_request_number:d}/head".format(
                pull_request_number=pull_request_number
            ),
        ]()
        .strip()
        .split()[0]
    )
    merge_pr(
        pull_request_number,
        base_ref_commit,
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
