# Style and formatting

## Code style

- [`black`](https://github.com/psf/black): Formatting Python code
- [`ruff`](https://github.com/charliermarsh/ruff): Formatting and sorting `import` statements
- [`shellcheck`](https://github.com/koalaman/shellcheck): Linting shell scripts
- [`shfmt`](https://github.com/mvdan/sh): Formatting shell scripts
- [`statix`](https://github.com/nerdypepper/statix): Linting nix files
- [`nixpkgs-fmt`](https://github.com/nix-community/nixpkgs-fmt): Formatting nix files

::: {.callout-tip}
If you use `nix-shell`, all of these are already setup for you and ready to use, and you don't need to do anything to install these tools.
:::

We use [numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html) as our
standard format for docstrings.
