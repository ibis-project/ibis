---
hide:
  - toc
---

# Setting Up a Development Environment

## Required Dependencies

- [`git`](https://git-scm.com/)

::: {.panel-tabset}

## pip

::: {.callout-warning}
`pip` will not handle installation of system dependencies

`pip` will not install system dependencies needed for some packages such as `psycopg2` and `kerberos`.

For a better development experience see the `conda` or `nix` setup instructions.
:::

1. [Install `gh`](https://cli.github.com/manual/installation)

1. Fork and clone the ibis repository:

    ```sh
    gh repo fork --clone --remote ibis-project/ibis
    ```

1. Change directory into `ibis`:

    ```sh
    cd ibis
    ```

1. Install development dependencies

    ```sh
    pip install 'poetry>=1.3,<1.4'
    pip install -r requirements-dev.txt
    ```

1. Install ibis in development mode

    ```sh
    pip install -e .
    ```

## Conda

::: {.callout-note}
Some optional dependencies for Windows are not available through `conda`/`mamba`
:::

1. `clickhouse-cityhash`. Required for compression support in the ClickHouse backend.

### Support Matrix

| Python Version | Python 3.9 | Python 3.10 | Python 3.11 |
| - | - | - | - |
| **Operating System** | | | |
|  **Linux** | {{ config.extra.support_levels.supported.icon }} | {{ config.extra.support_levels.supported.icon }} | {{ config.extra.support_levels.supported.icon }} |
| **macOS (x86_64)** | {{ config.extra.support_levels.supported.icon }} | {{ config.extra.support_levels.supported.icon }} | {{ config.extra.support_levels.supported.icon }} |
| **macOS (aarch64)** | {{ config.extra.support_levels.supported.icon }} | {{ config.extra.support_levels.supported.icon }} | {{ config.extra.support_levels.supported.icon }} |
| **Windows** | {{ config.extra.support_levels.supported.icon }} | {{ config.extra.support_levels.supported.icon }} | {{ config.extra.support_levels.supported.icon }} |

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

        {% set platforms = {"Linux": "linux-64", "macOS (x86_64)": "osx-64", "macOS (aarch64)": "osx-arm64", "Windows": "win-64"} %}
        {% for os, platform in platforms.items() %}
        === "{{ os }}"

            ```sh
            # Create a dev environment for {{platform}}
            cd ibis
            {{ manager }} create -n ibis-dev --file=conda-lock/{{ platform }}-3.10.lock
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

{% endfor %}


## Nix

#### Support Matrix

|      Python Version :material-arrow-right: |                     Python 3.9                     |                    Python 3.10                     |                    Python 3.11                     |
| -----------------------------------------: | :------------------------------------------------: | :------------------------------------------------: | :------------------------------------------------: |
| **Operating System** :material-arrow-down: |                                                    |                                                    |                                                    |
|                                  **Linux** |  {{ config.extra.support_levels.supported.icon }}  |  {{ config.extra.support_levels.supported.icon }}  |  {{ config.extra.support_levels.supported.icon }}  |
|                         **macOS (x86_64)** |  {{ config.extra.support_levels.supported.icon }}  |  {{ config.extra.support_levels.supported.icon }}  |  {{ config.extra.support_levels.supported.icon }}  |
|                        **macOS (aarch64)** |   {{ config.extra.support_levels.unknown.icon }}   |   {{ config.extra.support_levels.unknown.icon }}   |   {{ config.extra.support_levels.unknown.icon }}   |
|                                **Windows** | {{ config.extra.support_levels.unsupported.icon }} | {{ config.extra.support_levels.unsupported.icon }} | {{ config.extra.support_levels.unsupported.icon }} |

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

    This may take a while due to artifact download from the cache.

:::

## Building the Docs

Run

```bash
mkdocs serve --strict
```

to build and serve the documentation.
