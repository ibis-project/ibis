# Maintaining the Codebase

Ibis maintainers are expected to handle the following tasks as they arise:

- Reviewing and merging pull requests
- Triaging new issues

## Dependencies

A number of tasks that are typically associated with maintenance are partially or fully automated.

|             Dependency Type | Management Tool                                                                                                    |
| --------------------------: | :----------------------------------------------------------------------------------------------------------------- |
| Python library dependencies | [WhiteSource Renovate](https://www.whitesourcesoftware.com/free-developer-tools/renovate/)                         |
|              GitHub Actions | [WhiteSource Renovate](https://www.whitesourcesoftware.com/free-developer-tools/renovate/)                         |
|            Nix dependencies | [A GitHub Action](https://github.com/ibis-project/ibis/actions/workflows/update-deps.yml) run at a regular cadence |

Dependencies are managed using [`poetry`](https://python-poetry.org).

Occasionally you may need to lock poetry dependencies, which can be
done by running

```sh
poetry lock --no-update
```

### Automatic Dependency Updates

[WhiteSource
Renovate](https://www.whitesourcesoftware.com/free-developer-tools/renovate/)
will run at some cadence (outside of traditional business hours) and submit PRs
that update dependencies.

These upgrades use a conservative update strategy, which is currently to
increase the upper bound of a dependency's version range.

The PRs it generates will regenerate a number of other files so that in most
cases contributors do not have to remember to generate and commit these files.

### Adding or Changing Dependencies

1.  Edit `pyproject.toml` as needed.
2.  Run `poetry lock --no-update`

Updates of minor and patch versions of dependencies are handled automatically by
[`renovate`](https://github.com/renovatebot/renovate).

## Merging PRs

PRs can be merged using the [`gh` command line tool](https://github.com/cli/cli)
or with the GitHub web UI.

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
