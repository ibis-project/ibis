# Contributing to Ibis

## Set up a development environment

There are two primary ways to setup a development environment.

* [`nix`](#nix): fewer steps, isolated, complete environment, less familiar
* [`conda`](#miniconda): more steps, less isolated, more familiar

**NOTE:** that most of the development scripts require using nix but those scripts
are not necessary for most contributions and used almost exclusively in CI.

### Initial steps

**Dependencies:**

- required: `git`
- optional: `gh`
- optional: `nix`
- optional: `conda`

Create a fork of the [Ibis repository](https://github.com/ibis-project/ibis), and clone it.

     :::sh
     git clone https://github.com/<your-github-username>/ibis

For a command-line-only experience, the `gh` tool is useful:

     :::sh
     gh repo clone ibis-project/ibis
     cd ibis
     gh repo fork

### Nix

1. [Download and install `nix`](https://nixos.org/guides/install-nix.html)
2. Run `nix-shell` in the checkout directory:
   
        :::sh
        cd ibis

        # set up the cache to avoid building everything from scratch
        nix-shell -A build --run 'cachix use ibis'

        # this will put you inside of a shell that has everything you need for development
        # the first time you run this, it will download a lot of things, but subsequent
        # runs will be much faster to start.
        nix-shell -A python39

### Miniconda

1. [Download](https://docs.conda.io/en/latest/miniconda.html) and install Miniconda
2. [Download the latest `environment.yaml`](https://github.com/ibis-project/ibis/releases/latest/download/environment.yaml)
3. Create a Conda environment suitable for ibis development:

        :::sh
        cd ibis
        conda create -n ibis-dev -f conda-lock/<platform-64-pyver>.lock


4. Activate the environment

        :::sh
        conda activate ibis-dev

5. Install your local copy of Ibis into the Conda environment. In the root of the project run:

        :::sh
        pip install -e .

### General workflow

#### Find an issue to work on

If you are working with Ibis, and find a bug, or you are reading the documentation and see something
wrong, or that could be clearer, you can work on that.

But sometimes, you may want to contribute to Ibis, but you don't have anything in mind. In that case,
you can check the GitHub issue tracker for Ibis, and look for issues with the label
[good first issue](https://github.com/ibis-project/ibis/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22).
Feel free to also help with other issues that don't have the label, but they may be more challenging,
and require knowledge of Ibis internals.

Once you found an issue you want to work on, write a comment with the text `/take`, and GitHub will
assign the issue to yourself. This way, nobody else will work on it at the same time. If you find an
issue that someone else is assigned to, please contact the assignee to know if they are still working
on it.

#### Make a branch

**Dependencies:**

- required: `git`

Once you're inside of a nix-shell, the first thing you want to do is make a branch. Let's call it `useful-bugfix`.

       :::sh
       git checkout -b useful-bugfix

#### Make the desired change

**Dependencies:**

- required: `git`

Let's say you've made a change to `ibis/expr/types.py` to fix a bug reported in issue #9000 (a made up issue at the time of writing).

Running `git status` should give output similar to this:

        :::sh
        $ git status
        On branch useful-bugfix
        Your branch is up to date with 'origin/useful-bugfix'.
        
        Changes not staged for commit:
          (use "git add <file>..." to update what will be committed)
          (use "git restore <file>..." to discard changes in working directory)
                modified:   ibis/expr/types.py
        
        no changes added to commit (use "git add" and/or "git commit -a")

#### Run the test suite

**Dependencies:**

- required: [`nix`](#nix) environment or [`conda`](#miniconda) environment

To run Ibis tests use the following command:

```sh
PYTEST_BACKENDS="sqlite pandas" python -m pytest ibis/tests
```

You can change `sqlite pandas` to include one or more supported backends (space
separated) that you want to test.

It isn't necessary to provide `PYTEST_BACKENDS` at all, but it's useful for
exercising more of the library.

#### Commit and push your changes

**Dependencies:**

- required: `git`
- required: `cz` (available in the `nix` environment)

Next, you'll want to commit your changes. Ibis's commit message structure
follows [that of the Angular
community](https://github.com/angular/angular/blob/master/CONTRIBUTING.md#-commit-message-format),
for maximum automation and consistency no matter who's contributing.

**NOTE:** Ensure that you have run `pre-commit run` **before** running `cz`, otherwise
you may fail the pre-commit checks and `cz` will not remember what you've typed.

You'll want to use `cz` to make it easy to adhere to this structure without much pain.

`cz` is already configured and ready to go if you're in a `nix-shell`, so just stage your changes and run `cz`:

        :::sh
        $ git add .
        $ cz

You should see a series of (hopefully informative) prompts about what action to take next:

1. Select the type of change you're committing. In this case, we're committing a bug fix, so we'll select fix:

        :::sh
        ? Select the type of change that you're committing:
          feat:     A new feature
        ❯ fix:      A bug fix
          docs:     Documentation only changes
          style:    Changes that do not affect the meaning of the code (white-space, formatting, missing semi-colons, etc)
          refactor: A code change that neither fixes a bug nor adds a feature
          perf:     A code change that improves performance
          test:     Adding missing tests or correcting existing tests
        (Move up and down to reveal more choices) 

   Generally you don't need to think too hard about what category to select, but note that:

   * `feat` will cause a minor version bump
   * `fix` will cause a patch version bump
   * everything else will not cause a version bump, **unless it's a breaking
       change** (continue reading these instructions for more info on that)

2. Next, you're asked what the scope of this change is:

        :::sh
        ? What is the scope of this change (e.g. component or file name): (press enter to skip)

   This is optional, but if there's a clear component or single file that is
   modified you should put it. In our case, let's assume the bug fixed a type
   inference problem, so we'd type in `type-inference` at this prompt.

3. You'll then be asked to type in a short description of the change which will be the commit message title:

        :::sh
        ? Write a short, imperative tense description of the change (max 79 chars):
         (69) fix a type inference issue where floats were incorrectly cast to ints

   Let's say there was a problem with spurious casting of float to integers, so
   we type in the message above.  That number on the left (here `(69)`) is the
   length of description you've typed in.

4. Next you'll be asked for a longer description, which is entirely optional
   **unless the change is a breaking change**, or you feel like a bit of prose

        :::sh
        ? Provide a longer description of the change: (press enter to skip)
        A bug was triggered by some incorrect code that caused floats to be incorrectly cast to integers.

   For non breaking changes, this isn't strictly necessary but it can be very
   helpful when a change is large, obscure, or complex. For this example let's just reiterate
   most of what the commit title says.

5. Next you're asked about breaking changes:

        :::sh
        ? Are there any breaking changes? (y/N)

   If you answer `y`, then you'll get an additional prompt asking you to
   describe the breaking changes. This description will ultimately make its way
   into the user-facing release notes. If there aren't any breaking changes, press enter.
   Let's say this bug fix does **not** introduce a breaking change.

6. Finally, you're asked whether this change affects any open issues and if yes then to reference them:

        :::sh
        ? Does this change affect any open issues? (y/N) y
        ? Add issue references (e.g. "fix #123", "re #123".):
        fixes #9000

   Here we typed `fixes #9000` to indicate that we fixed issue #9000.

Whew! Seems like a lot, but it's rather quick once you get used to it. After
that you should have a commit that looks roughly like this, ready to be automatically rolled into the next release:

        commit e665cb960c7ba83493ce8d09dd51713c1c9e0d17 (HEAD -> nix-env)
        Author: Phillip Cloud <417981+cpcloud@users.noreply.github.com>
        Date:   Wed Sep 1 17:50:21 2021 -0400
        
            fix(type-inference): fix a type inference issue where floats were incorrectly cast to ints
        
            A bug was triggered by some incorrect code that caused floats to be incorrectly cast to integers.
        
            fixes #9000

### Updating dependencies

#### Automatic dependency updates

WhiteSource Renovate will run every day (outside of traditional business hours)
and submit PRs that attempt to update dependencies.

These upgrades use a very conservative update strategy, which is only to
**widen** the dependency range, as opposed to increasing the lower bound.

The PRs will also automatically regenerate `poetry.lock` and `setup.py`, so
that in most cases, maintainers and contributors do not have to remember to
generate and commit these files.

In case you need to manually update a dependency see the next section.

#### Manually updating dependencies

**Do not manually edit `setup.py`, it is automatically generated from `pyproject.toml`**

1. Edit `pyproject.toml` as needed.
2. Run `poetry update`
3. Run

```sh
$ PYTHONHASHSEED=42 python ./dev/poetry2setup.py -o setup.py
```

from the repository root. `tomli` and `poetry-core` are necessary to run this
script, they can be `pip install`ed.

4. Commit your changes and make a pull request.

Updates of minor versions of dependencies are handled automatically by
[`renovate`](https://github.com/renovatebot/renovate).

Do not make PRs from changes resulting from running `poetry update` unless
absolutely necessary.

### Releasing

#### Make a pull request

**Dependencies:**

- required: `gh` or a web browser

Making a pull request to the Ibis repository is very straightforward using the `gh` command line tool:

        :::sh
        $ gh pr create

Follow the prompts, and you should have a PR up in no time!

Alternatively, a PR can also be created through the web UI after pushing your
branch up to your remote (here, that's `origin`):

        :::sh
        $ git push --set-upstream origin useful-bugfix

## Working with backends

If you don't have a database for the backend you want to work on, you can check
the configuration of the continuos integration. There, docker is used to run a
few backends as services. These services are defined in
`.github/workflows/main.yml`.

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

More may be added in the future.

**Note:** If you use `nix-shell` all of these are setup for you and ready to use, you don't
need to manually install any of these tools.

We use [numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html) as our
standard format for docstrings.

## Commit philosophy

We aim to make our individual commits small and tightly focused on the feature
they are implementing or bug being fixed. If you find yourself making
functional changes to different areas of the codebase, we prefer you break up
your changes into separate Pull Requests. In general, a philosophy of one
Github Issue per Pull Request is a good rule of thumb, though that isn't always
possible.

We avoid merge commits (they are disabled in the Github repository) so you may
be asked to rebase your changes on top of the latest commits to master if there
have been changes since you last updated a Pull Request.  Rebasing your changes
is usually as simple as running ``git pull upstream master --rebase`` and then
force-pushing to your branch: ``git push origin <branch-name> -f``.

## Maintainer's guide

Maintainers generally perform a number of roles:

* Merging pull requests
* Making releases to conda-forge

A number of tasks that are typically associated with maintenance are either partially or fully automated:

* Updating library dependencies: this is handled automatically by dependabot
* Updating github-actions: this is handled automatically by dependabot
* Updating nix dependencies: this is a job run every six hours that will
  create pull request to update nix dependencies

Occasionally you may need to manually update poetry dependencies, which can be done by running

        :::sh
        nix-shell -A build --run 'poetry update'

or if `poetry` is already in `PATH`:

        :::sh
        poetry update

Assuming a dependency was actually updated, you'll see changes to `poetry.lock` in the current directory.

### Merging PRs

PRs can be merged using the [`gh` command line tool](https://github.com/cli/cli)
or with the GitHub web UI.

### Release

#### PyPI

Releases to PyPI are handled automatically using a [Python
implementation](https://python-semantic-release.readthedocs.io/en/latest/) of
[semantic
release](https://egghead.io/lessons/javascript-automating-releases-with-semantic-release).

In short, commit messages are examined for structure indicating various version-bumping changes.

If semantic release determines your changes are worthy of a release, it will execute all the necessary
steps, including updating version numbers in the right places and publishing artifacts to both PyPI
and GitHub.

Ibis is released in two places:

- [PyPI](https://pypi.org/) (the **PY**thon **P**ackage **I**ndex), to enable `pip install ibis-framework`
- [Conda Forge](https://conda-forge.org/), to enable `conda install ibis-framework`

#### `conda-forge`

The conda-forge package is released using the conda-forge feedstock repository: https://github.com/conda-forge/ibis-framework-feedstock/

After a release to PyPI, there'a conda recipe file that gets uploaded to the
GitHub release, which can be directly copy pasted into the conda-forge
feedstack and made into a pull request.

This is the recommended workflow for updating conda-forge's
ibis distribution. In the near future we hope to automate the PR submission as well.


#### Automated release deep dive

**NOTE:** this will likely be out of date, consult `.github/workflows/main.yml`
for up-to-date details

The release process consists of the following steps:

1. Wait on all non-release upstream jobs to finish, including linting, testing and conda package build testing.
1. If the workflow is a push to master, clone the repo with a token for automated pushing, otherwise clone without a token
1. Configure git username and email to indicate that github-actions will be making commits
1. Install nix
1. Setup cachix for nix caching
1. Run `poetry2setup` to generate a new `setup.py` that may have changed as a result of `pyproject.toml` changing
1. Commit the new file and push it to master **skippping CI**
1. Compute the hash of the new master from just pushing
1. Clone again if this workflow is running from a push to master to pick up the new changes
1. Install python using `actions/setup-python`
1. Install poetry and python-semantic-release
1. Run [`semantic-release publish`](https://python-semantic-release.readthedocs.io/en/latest/#semantic-release-publish)
   **using the `--noop` fiag if running from a pull request**
1. Store the commit hash of the release we just pushed

**NOTE:** the following steps are only run on a push to master

1. Clone the repo with the previously computed commit hash
1. Compute the URL of the ibis release tarball
1. Compute the SHA256 checksum of the tarball
1. Install nix
1. Setup cachix
1. Generate a conda recipe.yaml using `dev/poetry2recipe`
1. Upload the recipe to the release that just happened
1. Download the environment file that was produced during testing
1. Upload the environment file to the release that just happened
