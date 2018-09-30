#!/usr/bin/env python

import os
import random
import shutil
import sys
import tempfile

import click
import ruamel.yaml

from jinja2 import Environment, FileSystemLoader
from plumbum.cmd import git, conda

import ibis
from ibis.compat import Path, PY2


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


@cli.command()
@click.argument('repo-uri', default=default_repo)
@click.argument('destination', default=default_dest)
def clone(repo_uri, destination):
    if Path(destination).exists():
        return

    cmd = git['clone', repo_uri, destination]

    cmd(stdout=click.get_binary_stream('stdout'),
        stderr=click.get_binary_stream('stderr'))


SCRIPT = (
    '{{ PYTHON }} -m pip install . --no-deps --ignore-installed '
    '--no-cache-dir -vvv'
)


@cli.command()
@click.argument('meta',
                default=os.path.join(default_dest, 'recipe', 'meta.yaml'))
@click.option('--source-path', default=str(IBIS_DIR))
def update(meta, source_path):
    path = Path(meta)

    click.echo('Updating {} recipe...'.format(path.parent))

    content = render(path)
    recipe = ruamel.yaml.round_trip_load(content)

    # update the necessary fields, skip leading 'v' in the version
    recipe['package']['version'] = ibis.__version__[1:]
    recipe['source'] = {'path': source_path}

    # XXX: because render will remove the {{ PYTHON }} variable
    recipe['build']['script'] = SCRIPT

    updated_content = ruamel.yaml.round_trip_dump(
        recipe, default_flow_style=False, width=sys.maxsize).strip()

    if PY2:
        updated_content = updated_content.decode('utf-8')

    click.echo(updated_content)

    path.write_text(updated_content)


@cli.command()
@click.argument('recipe', default=os.path.join(default_dest, 'recipe'))
def build(recipe):
    click.echo('Building {} recipe...'.format(recipe))

    python_version = '{}.{}'.format(
        sys.version_info.major, sys.version_info.minor)

    cmd = conda['build', recipe,
                '--channel', 'conda-forge',
                '--python', python_version]

    cmd(stdout=click.get_binary_stream('stdout'),
        stderr=click.get_binary_stream('stderr'))


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
    cmd(stdout=click.get_binary_stream('stdout'),
        stderr=click.get_binary_stream('stderr'))


@cli.command()
@click.pass_context
def test(ctx):
    ctx.invoke(clone)
    ctx.invoke(update)
    ctx.invoke(build)
    ctx.invoke(deploy)


if __name__ == '__main__':
    cli()
