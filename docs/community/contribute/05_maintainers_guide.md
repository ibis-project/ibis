# Maintaining the Codebase

Ibis maintainers are expected to handle the following tasks as they arise:

- Reviewing and merging pull requests
- Triaging new issues

## Dependencies

A number of tasks that are typically associated with maintenance are partially or fully automated.

- [WhiteSource Renovate](https://www.whitesourcesoftware.com/free-developer-tools/renovate/) (Python library dependencies and GitHub Actions)
- [Custom GitHub Action](https://github.com/ibis-project/ibis/actions/workflows/update-deps.yml) (Nix dependencies)

### poetry

Occasionally you may need to lock [`poetry`](https://python-poetry.org) dependencies. Edit `pyproject.toml` as needed, then run:

```sh
poetry lock --no-update
```

## Release

Ibis is released on [PyPI](https://pypi.org/project/ibis-framework/) and [Conda Forge](https://github.com/conda-forge/ibis-framework-feedstock).

=== "PyPI"

    Releases to PyPI are handled automatically using [semantic
    release](https://egghead.io/lessons/javascript-automating-releases-with-semantic-release).

    To trigger a release use the [Release GitHub Action](https://github.com/ibis-project/ibis/actions/workflows/release.yml).

=== "`conda-forge`"

    The conda-forge package is maintained as a [conda-forge feedstock](https://github.com/conda-forge/ibis-framework-feedstock).

    After a release to PyPI, the conda-forge bot automatically updates the ibis
    package.
