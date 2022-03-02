---
hide:
  - toc
---

# Setting Up a Development Environment

## Required Dependencies

- [`git`](https://git-scm.com/)

!!! note "Python 3.10 is supported on a best-effort basis"

    As of 2022-02-17 there is support for Python 3.10 when using `nix` for development.

    `conda-forge` is still in [the process of migrating packages to Python
    3.10](https://conda-forge.org/status/#python310).

=== "Nix"

    #### Support Matrix

    |      Python Version :material-arrow-right: |                       Python 3.8                       |                     Python 3.9                     |                    Python 3.10                     |
    | -----------------------------------------: | :----------------------------------------------------: | :------------------------------------------------: | :------------------------------------------------: |
    | **Operating System** :material-arrow-down: |                                                        |                                                    |                                                    |
    |                                  **Linux** |  {{ config.extra.support_levels.supported.icon }}[^1]  |  {{ config.extra.support_levels.supported.icon }}  |  {{ config.extra.support_levels.supported.icon }}  |
    |                                  **macOS** |     {{ config.extra.support_levels.bug.icon }}[^2]     |     {{ config.extra.support_levels.bug.icon }}     |     {{ config.extra.support_levels.bug.icon }}     |
    |                                **Windows** | {{ config.extra.support_levels.unsupported.icon }}[^3] | {{ config.extra.support_levels.unsupported.icon }} | {{ config.extra.support_levels.unsupported.icon }} |

    1. [Install `nix`](https://nixos.org/download.html)
    1. Install `gh`:

        === "`nix-shell`"

            ```sh
            nix-shell -p gh
            ```

        === "`nix-env`"

            ```sh
            nix-env -iA gh
            ```

    1. Fork and clone the ibis repository:

        ```sh
        gh repo fork --clone --remote ibis-project/ibis
        ```

    1. Set up the public `ibis` Cachix cache to pull pre-built dependencies:

        ```sh
        nix-shell -p cachix --run 'cachix use ibis'
        ```

    1. Run `nix-shell` in the checkout directory:

        ```sh
        cd ibis
        nix-shell
        ```

        This may take awhile due to artifact download from the cache.

=== "Conda"

    !!! info "Some optional dependencies for Windows are not available through `conda`/`mamba`"

        1. `python-duckdb` and `duckdb-engine`. Required for the DuckDB backend.
        1. `clickhouse-cityhash`. Required for compression support in the ClickHouse backend.

    #### Support Matrix

    |      Python Version :material-arrow-right: |                      Python 3.8                      |                      Python 3.9                  |                  Python 3.10                   |
    | -----------------------------------------: | :--------------------------------------------------: | :----------------------------------------------: | :--------------------------------------------: |
    | **Operating System** :material-arrow-down: |                                                      |                                                  |                                                |
    |                                  **Linux** | {{ config.extra.support_levels.supported.icon }}[^1] | {{ config.extra.support_levels.supported.icon }} | {{ config.extra.support_levels.bug.icon }}[^2] |
    |                                  **macOS** |   {{ config.extra.support_levels.supported.icon }}   | {{ config.extra.support_levels.supported.icon }} |   {{ config.extra.support_levels.bug.icon }}   |
    |                                **Windows** |   {{ config.extra.support_levels.supported.icon }}   | {{ config.extra.support_levels.supported.icon }} |   {{ config.extra.support_levels.bug.icon }}   |

    {% set managers = {"conda": {"name": "Miniconda", "url": "https://docs.conda.io/en/latest/miniconda.html"}, "mamba": {"name": "Mamba", "url": "https://github.com/mamba-org/mamba"}} %}
    {% for manager, params in managers.items() %}

    === "`{{ manager }}`"

        1. Install [{{ params["name"] }}]({{ params["url"] }})

        1. Install `gh`

            ```sh
            {{ manager }} install -c conda-forge gh
            ```

        1. Fork and clone the ibis repository:

            ```sh
            gh repo fork --clone --remote ibis-project/ibis
            ```

        1. Create a Conda environment from a lock file in the repo:

            {% set platforms = {"Linux": "linux", "MacOS": "osx", "Windows": "win"} %}
            {% for os, platform in platforms.items() %}
            === "{{ os }}"

                ```sh
                cd ibis
                {{ manager }} create -n ibis-dev --file=conda-lock/{{ platform }}-64-3.9.lock
                ```
            {% endfor %}

        1. Activate the environment

            ```sh
            {{ manager }} activate ibis-dev
            ```

        1. Install your local copy of `ibis` into the Conda environment.

            ```sh
            cd ibis
            pip install -e .
            ```

        1. If you want to run the backend test suite you'll need to install `docker-compose`:

            ```sh
            {{ manager }} install docker-compose -c conda-forge
            ```

    {% endfor %}

Once you've set up an environment, try building the documentation:

```sh
mkdocs serve
```

{% for data in config.extra.support_levels.values() %}
[^{{ loop.index }}]: {{ data.description }}
{% endfor %}
