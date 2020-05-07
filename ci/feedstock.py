#!/usr/bin/env python

import os
import shutil
import sys
import tempfile
from pathlib import Path

import click
import ruamel.yaml
from jinja2 import Environment, FileSystemLoader
from plumbum.cmd import conda, git

import ibis

IBIS_DIR = Path(__file__).parent.parent.absolute()


def render(path):
    parent = str(path.parent)
    env = Environment(loader=FileSystemLoader(parent))
    template = env.get_template(path.name, parent=parent)
    return template.render()


@click.group()
def cli():
    pass


default_repo = 'https://github.com/conda-forge/ibis-framework-feedstock'
default_dest = os.path.join(tempfile.gettempdir(), 'ibis-framework-feedstock')
default_branch = 'master'


@cli.command()
@click.argument('repo-uri', default=default_repo)
@click.argument('destination', default=default_dest)
@click.option('-b', '--branch', default=default_branch)
def clone(repo_uri, destination, branch):
    if not Path(destination).exists():
        cmd = git['clone', repo_uri, destination]
        cmd(
            stdout=click.get_binary_stream('stdout'),
            stderr=click.get_binary_stream('stderr'),
        )

    cmd = git['-C', destination, 'checkout', branch]
    cmd(
        stdout=click.get_binary_stream('stdout'),
        stderr=click.get_binary_stream('stderr'),
    )


SCRIPT = (
    '{{ PYTHON }} -m pip install . --no-deps --ignore-installed '
    '--no-cache-dir -vvv'
)


@cli.command()
@click.argument(
    'meta', default=os.path.join(default_dest, 'recipe', 'meta.yaml')
)
@click.option('--source-path', default=str(IBIS_DIR))
def update(meta, source_path):
    path = Path(meta)

    click.echo('Updating {} recipe...'.format(path.parent))

    content = render(path)
    recipe = ruamel.yaml.round_trip_load(content)

    # update the necessary fields, skip leading 'v' in the version
    version = ibis.__version__
    assert not version.startswith('v'), 'version == {}'.format(version)
    recipe['package']['version'] = version
    recipe['source'] = {'path': source_path}

    # XXX: because render will remove the {{ PYTHON }} variable
    recipe['build']['script'] = SCRIPT

    updated_content = ruamel.yaml.round_trip_dump(
        recipe, default_flow_style=False, width=sys.maxsize
    ).strip()

    click.echo(updated_content)

    path.write_text(updated_content)


@cli.command()
@click.argument('recipe', default=os.path.join(default_dest, 'recipe'))
@click.option(
    '--python',
    default='{}.{}'.format(sys.version_info.major, sys.version_info.minor),
)
def build(recipe, python):
    click.echo('Building {} recipe...'.format(recipe))

    cmd = conda[
        'build', '--channel', 'conda-forge', '--python', python, recipe
    ]

    cmd(
        stdout=click.get_binary_stream('stdout'),
        stderr=click.get_binary_stream('stderr'),
    )


@cli.command()
@click.argument('package_location', default='/opt/conda/conda-bld')
@click.argument('artifact_directory', default='/tmp/packages')
@click.argument('architecture', default='linux-64')
def deploy(package_location, artifact_directory, architecture):
    artifact_dir = Path(artifact_directory)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    package_loc = Path(package_location)
    assert package_loc.exists(), 'Path {} does not exist'.format(package_loc)

    for architecture in (architecture, 'noarch'):
        arch_artifact_directory = str(artifact_dir / architecture)
        arch_package_directory = str(package_loc / architecture)
        shutil.copytree(arch_package_directory, arch_artifact_directory)
    cmd = conda['index', artifact_directory]
    cmd(
        stdout=click.get_binary_stream('stdout'),
        stderr=click.get_binary_stream('stderr'),
    )


@cli.command()
@click.pass_context
@click.option(
    '--python',
    default='{}.{}'.format(sys.version_info.major, sys.version_info.minor),
)
def test(ctx, python):
    ctx.invoke(clone)
    ctx.invoke(update)
    ctx.invoke(build, python=python)
    ctx.invoke(deploy)


if __name__ == '__main__':
    cli()
