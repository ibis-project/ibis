#!/usr/bin/env python

"""Generate release notes for Ibis."""

import datetime
import pathlib

import click
import pygit2
import regex as re

GITHUB_CLOSE_KEYWORDS = (
    'close',
    'closes',
    'closed',
    'fix',
    'fixes',
    'fixed',
    'resolve',
    'resolves',
    'resolved',
)

KEYWORD_MAP = {
    'bld': 'support',
    'build': 'support',
    'bug': 'bug',
    'ci': 'support',
    'doc': 'support',
    'enhance': 'feature',
    'enh': 'feature',
    'feat': 'feature',
    'feature': 'feature',
    'supp': 'support',
    'test': 'support',
    'tst': 'support',
}


def commits_between(repo, start_ref, end_ref, options=None):
    """Yield commits from `repo` between `start_ref` and `end_ref`."""
    start = repo.revparse_single(start_ref)
    end = repo.revparse_single(end_ref)
    for commit in repo.walk(end.oid, options):
        if commit.oid == start.oid:
            return
        else:
            yield commit


def iter_release_notes(repo, from_ref, to_ref, default_role):
    """Yield release notes from `from_ref` to `to_ref`."""
    pattern = re.compile(
        r'^(?:{})\s+#(\d+)\s+from'.format('|'.join(GITHUB_CLOSE_KEYWORDS)),
        flags=re.MULTILINE | re.IGNORECASE,
    )
    for commit in commits_between(
        repo, from_ref, to_ref, options=pygit2.GIT_SORT_TOPOLOGICAL
    ):
        message = commit.message.strip()
        subject, *lines = map(str.strip, message.splitlines())
        tag, *rest = subject.split(':', 1)
        tag = tag.lower()
        lineitem = ''.join(rest) or subject
        role = KEYWORD_MAP.get(tag, default_role)
        modifier = ' major' if role == 'bug' else ''
        try:
            issue_number, *_ = pattern.findall(message)
        except ValueError:
            issue_number = '-'
        yield "* :{role}:`{issue_number}{modifier}` {lineitem}".format(
            role=role,
            issue_number=issue_number,
            modifier=modifier,
            lineitem=lineitem.strip(),
        )


repo_path = pathlib.Path(__file__).parent.parent
repo = pygit2.Repository(str(repo_path))


@click.command()
@click.argument('release_version')
@click.option(
    '-f',
    '--from',
    'from_',
    default=repo.describe(
        describe_strategy=pygit2.GIT_DESCRIBE_TAGS, abbreviated_size=0
    ),
    help=(
        "The reference from which to calculate release notes. Defaults to "
        "the most recent tag."
    ),
    show_default=True,
)
@click.option(
    '-t',
    '--to',
    default="upstream/master",
    help="The last reference to include in release notes.",
    show_default=True,
)
@click.option(
    '-d',
    '--release-date',
    type=str,
    default=datetime.datetime.now().date().strftime('%Y-%m-%d'),
    help="The date of the release. Defaults to the current date.",
    show_default=True,
)
@click.option(
    '-r',
    '--default-role',
    default='support',
    help=(
        "The Sphinx role to use if a known prefix is not found in a "
        "commit's subject line."
    ),
    show_default=True,
)
def main(release_version, from_, to, release_date, default_role):
    title = "* :release:`{release} {date}`".format(
        release=release_version, date=release_date
    )
    click.echo(title)
    click.echo('\n'.join(iter_release_notes(repo, from_, to, default_role)))


if __name__ == '__main__':
    main()
