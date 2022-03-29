# Style and Formatting

## Code Style

The following tools are run in both CI and `pre-commit` checks to ensure codebase hygiene:

|                                                                   Tool | Purpose                                             |
| ---------------------------------------------------------------------: | :-------------------------------------------------- |
|                                [`black`](https://github.com/psf/black) | Formatting Python code                              |
|                              [`isort`](https://github.com/PyCQA/isort) | Formatting and sorting `import` statements          |
| [`absolufy-imports`](https://github.com/MarcoGorelli/absolufy-imports) | Automatically convert relative imports to absolute. |
|                        [`flake8`](https://flake8.pycqa.org/en/latest/) | Linting Python code                                 |
|              [`nix-linter`](https://github.com/Synthetica9/nix-linter) | Linting nix files                                   |
|          [`nixpkgs-fmt`](https://github.com/nix-community/nixpkgs-fmt) | Formatting nix files                                |
|                 [`shellcheck`](https://github.com/koalaman/shellcheck) | Linting shell scripts                               |
|                                 [`shfmt`](https://github.com/mvdan/sh) | Formatting shell scripts                            |
|                   [`pyupgrade`](https://github.com/asottile/pyupgrade) | Ensuring the latest available Python syntax is used |

!!! tip

    If you use `nix-shell` all of these are setup for you and ready to use, you don't
    need to install any of these tools.

We use [numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html) as our
standard format for docstrings.

## Commit philosophy

We aim to make our individual commits small and tightly focused on the feature
they are implementing or bug being fixed. If you find yourself making
functional changes to different areas of the codebase, we prefer you break up
your changes into separate Pull Requests. In general, a philosophy of one
Github Issue per Pull Request is a good rule of thumb.
