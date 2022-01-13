# Setting up a Development Environment

There are two primary ways to setup a development environment.

- `nix`: fewer steps, isolated
- `conda`: more steps, not isolated

## Required Dependencies

- [`git`](https://git-scm.com/)

## Package Management

!!! warning "You need at least one package manager"

    At least one of `nix` or `conda` is required to contribute to ibis.

!!! info "Python 3.10 is supported on a best-effort basis"

    As of 2022-01-05 there is *experimental* support for Python 3.10.
    However, there are a number of problems with dependencies and development
    tools that ibis uses and we cannot offically support Python 3.10 until
    those are fixed.

=== "Nix (Linux, Python 3.8-3.9)"

    1. [Download and install `nix`](https://nixos.org/download.html)
    1. Install `gh`:
        ```sh
        nix-shell -p gh
        # or
        nix-env -iA gh
        ```

    1. Fork and clone the ibis repository:

        ```sh
        # you will likely need to auth, gh will guide you through the steps
        gh repo fork --clone --remote ibis-project/ibis
        ```

    1. Run `nix-shell` in the checkout directory:

        ```sh
        cd ibis

        # set up the cache to avoid building everything from scratch
        nix-shell -p cachix --run 'cachix use ibis'

        # start a nix-shell
        #
        # this may take awhile to download artifacts from the cache
        nix-shell
        ```

=== "Miniconda (Linux, Mac, Windows, Python 3.8-3.9)"

    !!! tip "Mamba is supported as well"

        [Mamba](https://github.com/mamba-org/mamba) and [Mambaforge](https://github.com/conda-forge/miniforge#mambaforge) can be used in place of `conda`.

    1. [Download](https://docs.conda.io/en/latest/miniconda.html) and install Miniconda

    1. Install `gh`

        ```sh
        conda install -c conda-forge gh
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
            conda create -n ibis-dev --file=conda-lock/{{ platform }}-64-3.9.lock
            ```
        {% endfor %}

    1. Activate the environment

        ```sh
        conda activate ibis-dev
        ```

    1. Install your local copy of `ibis` into the Conda environment. In the root of the project run:

            pip install -e .

    1. If you want to run the backend test suite you'll need to install `docker-compose`:

            conda install docker-compose -c conda-forge
