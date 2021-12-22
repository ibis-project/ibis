# Contributing to Ibis

## Set up a development environment

There are two primary ways to setup a development environment.

* [`nix`](#nix): fewer steps, isolated
* [`conda`](#miniconda): more steps, not isolated

### Initial steps

**Dependencies:**

- required: [`git`](https://git-scm.com/)
- required: [`gh`](https://github.com/cli/cli)
- optional: [`nix`](https://nixos.org/download.html#nix-quick-install)
- optional: [`conda`](https://docs.conda.io/en/latest/)

Installing both `nix` and `conda` is fine, but you should use one or the other for contributing, not both.

Use `gh` to fork and clone the `ibis-project/ibis` repository:

        gh repo fork --clone --remote ibis-project/ibis
        cd ibis

### Nix

1. [Download and install `nix`](https://nixos.org/guides/install-nix.html)
2. Run `nix-shell` in the checkout directory:
   
        cd ibis

        # set up the cache to avoid building everything from scratch
        nix-shell -p cachix --run 'cachix use ibis'

        # start a nix-shell
        #
        # this may take awhile to download artifacts from the cache
        nix-shell

### Miniconda

1. [Download](https://docs.conda.io/en/latest/miniconda.html) and install Miniconda
2. [Download the latest `environment.yaml`](https://github.com/ibis-project/ibis/releases/latest/download/environment.yaml)
3. Create a Conda environment suitable for ibis development:

        cd ibis
        conda create -n ibis-dev -f conda-lock/<platform-64-pyver>.lock


4. Activate the environment

        conda activate ibis-dev

5. Install your local copy of `ibis` into the Conda environment. In the root of the project run:

        pip install -e .

### General workflow

#### Find an issue to work on

All contributions are welcome! Code, docs, and constructive feedback are all
great contributions to the project.

If you don't have a particular issue in mind head over to the GitHub issue
tracker for Ibis and look for open issues with the label [`good first
issue`](https://github.com/ibis-project/ibis/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22).

Feel free to help with other issues that aren't labeled as such, but they may be more challenging.

Once you find an issue you want to work on, write a comment with the text
`/take` on the issue. GitHub will then assign the issue to you.

This lets people know you're working on the issue. If you find an issue that
has an assignee, comment on the issue and ask whether the assignee is still
working on the issue.

#### Make a branch

**Dependencies:**

- required: `git`

The first thing you want to do is make a branch. Let's call it `useful-bugfix`.

        git checkout -b useful-bugfix

#### Make the desired change

**Dependencies:**

- required: `git`

Let's say you've made a change to `ibis/expr/types.py` to fix a bug reported in issue #424242 (not actually an issue).

Running `git status` should give output similar to this:

        $ git status
        On branch useful-bugfix
        Your branch is up to date with 'origin/useful-bugfix'.
        
        Changes not staged for commit:
          (use "git add <file>..." to update what will be committed)
          (use "git restore <file>..." to discard changes in working directory)
                modified:   ibis/expr/types.py
        
        no changes added to commit (use "git add" and/or "git commit -a")

#### Run the test suite

Next, you'll want to run a subset of the test suite.

**Dependencies:**

- required: [`nix`](#nix) environment or [`conda`](#miniconda) environment

To run a subset of the ibis tests use the following command:

```sh
PYTEST_BACKENDS="sqlite pandas" pytest ibis/tests ibis/backends/tests
```

You can change `"sqlite pandas"` to include one or more space-separated
supported backends that you want to test.

It isn't necessary to provide `PYTEST_BACKENDS` at all, but it's useful for
exercising more of the library's test suite.

#### Commit your changes

**Dependencies:**

- required: `git`
- optional: [`cz`](https://commitizen-tools.github.io/commitizen/)

Next, you'll want to commit your changes.

Ibis's commit message structure follows [`semantic-release`
conventions](https://github.com/semantic-release/semantic-release).

**NOTE:** It isn't necessary to use `cz commit` to make commits, but it is
necessary to follow outlined in this table in [this
table](https://github.com/semantic-release/semantic-release).

`cz` is already configured and ready to go if you've setup an environment, so
stage your changes and run `cz commit`:

        git add .
        cz commit

You should see a series of prompts about actions to take next:

1. Select the type of change you're committing. In this case, we're committing a bug fix, so we'll select fix:

        ? Select the type of change you are committing (Use arrow keys)
         » fix: A bug fix. Correlates with PATCH in SemVer
           feat: A new feature. Correlates with MINOR in SemVer
           docs: Documentation only changes
           style: Changes that do not affect the meaning of the code (white-space, formatting, missing semi-colons, etc)
           refactor: A code change that neither fixes a bug nor adds a feature
           perf: A code change that improves performance
           test: Adding missing or correcting existing tests
           build: Changes that affect the build system or external dependencies (example scopes: pip, docker, npm)
           ci: Changes to our CI configuration files and scripts (example scopes: GitLabCI)

   Generally you don't need to think too hard about what category to select, but note that:

   * `feat` will cause a minor version bump
   * `fix` will cause a patch version bump
   * everything else will not cause a version bump, **unless it's a breaking
       change** (continue reading these instructions for more info on that)

2. Next, you're asked what the scope of this change is:

        ? What is the scope of this change? (class or file name): (press [enter] to skip)

   This is optional, but if there's a clear component or single file that is
   modified you should put it. In our case, let's assume the bug fixed a type
   inference problem, so we'd type in `type-inference` at this prompt.

3. You'll then be asked to type in a short description of the change which will be the commit message title:

        ? Write a short and imperative summary of the code changes: (lower case and no period)
         fix a type inference issue where floats were incorrectly cast to ints

   Let's say there was a problem with spurious casting of float to integers, so
   we type in the message above.  That number on the left (here `(69)`) is the
   length of description you've typed in.

4. Next you'll be asked for a longer description, which is entirely optional
   **unless the change is a breaking change**, or you feel like a bit of prose

        ? Provide additional contextual information about the code changes: (press [enter] to skip)
         A bug was triggered by some incorrect code that caused floats to be incorrectly cast to integers.

   For non breaking changes, this isn't strictly necessary but it can be very
   helpful when a change is large, obscure, or complex. For this example let's just reiterate
   most of what the commit title says.

5. Next you're asked about breaking changes:

        ? Is this a BREAKING CHANGE? Correlates with MAJOR in SemVer (y/N)

   If you answer `y`, then you'll get an additional prompt asking you to
   describe the breaking changes. This description will ultimately make its way
   into the user-facing release notes. If there aren't any breaking changes, press enter.
   Let's say this bug fix does **not** introduce a breaking change.

6. Finally, you're asked whether this change affects any open issues (ignore
   the bit about breaking changes) and if yes then to reference them:

        ? Footer. Information about Breaking Changes and reference issues that this commit closes: (press [enter] to skip)
         fixes #424242

   Here we typed `fixes #424242` to indicate that we fixed issue #9000.

Whew! Seems like a lot, but it's rather quick once you get used to it. After
that you should have a commit that looks roughly like this, ready to be automatically rolled into the next release:

        commit 4049adbd66b0df48e37ca105da0b9139101a1318 (HEAD -> useful-bugfix)
        Author: Phillip Cloud <417981+cpcloud@users.noreply.github.com>
        Date:   Tue Dec 21 10:30:50 2021 -0500

            fix(type-inference): fix a type inference issue where floats were incorrectly cast to ints

            A bug was triggered by some incorrect code that caused floats to be incorrectly cast to integers.

            fixes #424242

#### Push your changes

Now that you've got a commit, you're ready to push your changes and make a pull request!

        $ gh pr create

Follow the prompts, and `gh` will print a link to your PR upon successfuly submission.

### Updating dependencies

#### Automatic dependency updates

[WhiteSource
Renovate](https://www.whitesourcesoftware.com/free-developer-tools/renovate/)
will run at some cadence (outside of traditional business hours) and submit PRs
that update dependencies.

These upgrades use a conservative update strategy, which is currently to
increase the upper bound of a dependency's range.

The PRs it generates will regenerate a number of other files so that in most
cases contributors do not have to remember to generate and commit these files.

#### Manually updating dependencies

**Do not manually edit `setup.py`, it is automatically generated from `pyproject.toml`**

1. Edit `pyproject.toml` as needed.
2. Run `poetry update`
3. Run

```sh
# if using nix
$ ./dev/poetry2setup -o setup.py

# it not using nix, requires installation of tomli and poetry-core
$ PYTHONHASHSEED=42 python ./dev/poetry2setup.py -o setup.py
```

from the repository root.

Updates of minor and patch versions of dependencies are handled automatically by
[`renovate`](https://github.com/renovatebot/renovate).

### Releasing

## Style and formatting

The following tools are run in both CI and pre-commit checks to check various
kinds of code style and lint rules:

- [black](https://github.com/psf/black) for general formatting of Python code
- [isort](https://github.com/PyCQA/isort) for formatting and sorting Python `import` statements
- [flake8](https://flake8.pycqa.org/en/latest/) for linting Python code
- [nix-linter](https://github.com/Synthetica9/nix-linter) for linting nix files
- [nixpkgs-fmt](https://github.com/nix-community/nixpkgs-fmt) for formatting nix files
- [prettier](https://prettier.io/) for formatting handwritten TOML, YAML, and JSON files in the repo
- [shellcheck](https://github.com/koalaman/shellcheck) for linting various shell script things
- [shfmt](https://github.com/mvdan/sh) for formatting shell scripts

**Note:** If you use `nix-shell` all of these are setup for you and ready to use, you don't
need to install any of these tools.

We use [numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html) as our
standard format for docstrings.

## Commit philosophy

We aim to make our individual commits small and tightly focused on the feature
they are implementing or bug being fixed. If you find yourself making
functional changes to different areas of the codebase, we prefer you break up
your changes into separate Pull Requests. In general, a philosophy of one
Github Issue per Pull Request is a good rule of thumb.

## Maintainer's guide

Maintainers should be performing a minimum number of tasks, deferring to automation
as much as possible:

* Merging pull requests
* Making releases to conda-forge

A number of tasks that are typically associated with maintenance are either partially or fully automated:

* Updating library dependencies: this is handled automatically by WhiteSource Renovate
* Updating github-actions: this is handled automatically by WhiteSource Renovate
* Updating nix dependencies: this is a job run at a regular cadence to update nix dependencies
  and publish them to a public cache

Occasionally you may need to manually lock poetry dependencies, which can be done by running

        poetry update --lock

If a dependency was updated, you'll see changes to `poetry.lock` in the current directory.

### Merging PRs

PRs can be merged using the [`gh` command line tool](https://github.com/cli/cli)
or with the GitHub web UI.

### Release

#### PyPI

Releases to PyPI are handled automatically using a [Python
implementation](https://python-semantic-release.readthedocs.io/en/latest/) of
[semantic
release](https://egghead.io/lessons/javascript-automating-releases-with-semantic-release).

Ibis is released in two places:

- [PyPI](https://pypi.org/) (the **PY**thon **P**ackage **I**ndex), to enable `pip install ibis-framework`
- [Conda Forge](https://conda-forge.org/), to enable `conda install ibis-framework`

#### `conda-forge`

The conda-forge package is released using the conda-forge feedstock repository: https://github.com/conda-forge/ibis-framework-feedstock/

After a release to PyPI, the conda-forge bot automatically update the ibis
package.
