# Style and Formatting

## Code Style

The following tools are run in both CI and `pre-commit` checks to ensure codebase hygiene:

- [`black`](https://github.com/psf/black): Formatting Python code
- [`isort`](https://github.com/PyCQA/isort): Formatting and sorting `import` statements
- [`absolufy-imports`](https://github.com/MarcoGorelli/absolufy-imports): Automatically convert relative imports to absolute
- [`flake8`](https://flake8.pycqa.org/en/latest/): Linting Python code
- [`nix-linter`](https://github.com/Synthetica9/nix-linter): Linting nix files
- [`nixpkgs-fmt`](https://github.com/nix-community/nixpkgs-fmt): Formatting nix files
- [`shellcheck`](https://github.com/koalaman/shellcheck): Linting shell scripts
- [`shfmt`](https://github.com/mvdan/sh): Formatting shell scripts
- [`pyupgrade`](https://github.com/asottile/pyupgrade): Ensuring the latest available Python syntax is used

!!! tip

    If you use `nix-shell`, all of these are already setup for you and ready to use, and you don't need to do anything to install these tools.

We use [numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html) as our
standard format for docstrings.
