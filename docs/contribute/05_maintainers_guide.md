# Maintaining the codebase

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

## Adding examples

If you're not a maintainer, please open an issue asking us to add your example.

### Requirements

You need the ability to write to the `gs://ibis-examples` GCS bucket to add an example.

### Instructions

Make sure you're in the root of the ibis git repository.

Assuming your file is called `example.csv`:

1. Add a gzip-compressed CSV file with the path `ibis/examples/data/example.csv.gz`.
1. Add a file named `ibis/examples/descriptions/example` that contains a
   description of your example. One line is best, but not necessary.
1. Run one of the following **from the git root of an ibis clone**:
   - `python ibis/examples/gen_registry.py` (doesn't include R dependenices)
   - `nix run '.#gen-examples'` (includes R dependenices)

## Release

Ibis is released on [PyPI](https://pypi.org/project/ibis-framework/) and [Conda Forge](https://github.com/conda-forge/ibis-framework-feedstock).

::: {.panel-tabset}

## PyPI

Releases to PyPI are handled automatically using [semantic release](https://egghead.io/lessons/javascript-automating-releases-with-semantic-release).

To trigger a release use the [Release GitHub Action](https://github.com/ibis-project/ibis/actions/workflows/release.yml).

## `conda-forge`

The conda-forge package is maintained as a [conda-forge feedstock](https://github.com/conda-forge/ibis-framework-feedstock).

After a release to PyPI, the conda-forge bot automatically updates the ibis package.

:::
