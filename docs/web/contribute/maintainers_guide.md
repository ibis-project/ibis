# Maintaining the Codebase

Maintainers should be performing a minimum number of tasks, deferring to automation
as much as possible:

- Reviewing pull requests
- Merging pull requests

A number of tasks that are typically associated with maintenance are partially
or fully automated:

- Updating library dependencies: this is handled automatically by WhiteSource Renovate
- Updating github-actions: this is handled automatically by WhiteSource Renovate
- Updating nix dependencies: this is a job run at a regular cadence to update nix dependencies

## Updating dependencies

Occasionally you may need to manually lock poetry dependencies, which can be done by running

```sh
poetry update --lock
```

If a dependency was updated, you'll see changes to `poetry.lock` in the current directory.

### Automatic dependency updates

[WhiteSource
Renovate](https://www.whitesourcesoftware.com/free-developer-tools/renovate/)
will run at some cadence (outside of traditional business hours) and submit PRs
that update dependencies.

These upgrades use a conservative update strategy, which is currently to
increase the upper bound of a dependency's range.

The PRs it generates will regenerate a number of other files so that in most
cases contributors do not have to remember to generate and commit these files.

### Manually updating dependencies

!!! danger

    Do not manually edit `setup.py`, it is automatically generated from `pyproject.toml`

1. Edit `pyproject.toml` as needed.
2. Run `poetry update`
3. Run

```sh
# if using nix
./dev/poetry2setup -o setup.py

# it not using nix, requires installation of tomli and poetry-core
PYTHONHASHSEED=42 python ./dev/poetry2setup.py -o setup.py
```

from the repository root.

Updates of minor and patch versions of dependencies are handled automatically by
[`renovate`](https://github.com/renovatebot/renovate).

## Merging PRs

PRs can be merged using the [`gh` command line tool](https://github.com/cli/cli)
or with the GitHub web UI.

## Release

### PyPI

Releases to PyPI are handled automatically using [semantic
release](https://egghead.io/lessons/javascript-automating-releases-with-semantic-release).

Ibis is released in two places:

- [PyPI](https://pypi.org/), to enable `pip install ibis-framework`
- [Conda Forge](https://conda-forge.org/), to enable `mamba install ibis-framework`

### `conda-forge`

The conda-forge package is released using the [conda-forge feedstock repository](https://github.com/conda-forge/ibis-framework-feedstock)

After a release to PyPI, the conda-forge bot automatically updates the ibis
package.
