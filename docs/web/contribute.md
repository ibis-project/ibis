# Contributing to Ibis

## Set up a development environment

1. Create a fork of the [Ibis repository](https://github.com/ibis-project/ibis), and clone it.

        :::sh
        git clone https://github.com/<your-github-username>/ibis


2. [Download](https://docs.conda.io/en/latest/miniconda.html) and install Miniconda
3. Create a Conda environment suitable for ibis development:
   
        :::sh
        cd  ibis
        conda env create


4. Activate the environment

        :::sh
        conda activate ibis-dev

5. Install your local copy of Ibis into the Conda environment. In the root of the project run:

        :::sh
        pip install -e .


## Find an issue to work on

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


## Working with backends

Ibis comes with several backends. If you want to work with a specific backend, you will have to install
the dependencies for the backend with `conda install -n ibis-dev -c conda-forge --file="ci/deps/<backend>.yml"`.

If you don't have a database for the backend you want to work on, you can check the configuration of the
continuos integration, where docker images are used for different backend. This is defined in
`.github/workflows/main.yml`.

## Run the test suite

To run Ibis tests use the next command:

```sh
PYTEST_BACKENDS="sqlite pandas" python -m pytest ibis/tests
```

You can change `sqlite pandas` by the backend or backends (space separated) that
you want to test.


## Style and formatting

We use [flake8](http://flake8.pycqa.org/en/latest/),
[black](https://github.com/psf/black) and
[isort](https://github.com/pre-commit/mirrors-isort) to ensure our code
is formatted and linted properly. If you have properly set up your development
environment by running ``make develop``, the pre-commit hooks should check
that your proposed changes continue to conform to our style guide.

We use [numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html) as
our standard format for docstrings.


## Commit philosophy

We aim to make our individual commits small and tightly focused on the feature
they are implementing. If you find yourself making functional changes to
different areas of the codebase, we prefer you break up your changes into
separate Pull Requests. In general, a philosophy of one Github Issue per
Pull Request is a good rule of thumb, though that isn't always possible.

We avoid merge commits (and in fact they are disabled in the Github repository)
so you may be asked to rebase your changes on top of the latest commits to
master if there have been changes since you last updated a Pull Request.
Rebasing your changes is usually as simple as running
``git pull upstream master --rebase`` and then force-pushing to your branch:
``git push origin <branch-name> -f``.


## Commit/PR messages

Well-structed commit messages allow us to generate comprehensive release notes
and make it very easy to understand what a commit/PR contributes to our
codebase. Commit messages and PR titles should be prefixed with a standard
code the states what kind of change it is. They fall broadly into 3 categories:
``FEAT (feature)``, ``BUG (bug)``, and ``SUPP (support)``. The ``SUPP``
category has some more fine-grained aliases that you can use, such as ``BLD``
(build), ``CI`` (continuous integration), ``DOC`` (documentation), ``TST``
(testing), and ``RLS`` (releases).


## Maintainer's guide

Maintainers generally perform two roles, merging PRs and making official
releases.


### Merging PRs

We have a CLI script that will merge Pull Requests automatically once they have
been reviewed and approved. See the help message in ``dev/merge-pr.py`` for
full details. If you have two-factor authentication turned on in Github, you
will have to generate an application-specific password by following this
[guide](https://help.github.com/en/articles/creating-a-personal-access-token-for-the-command-line).
You will then use that generated password on the command line for the ``-P``
argument.

Access the [Ibis "Merging PRs" wiki](https://github.com/ibis-project/ibis/wiki/Merging-PRs) page
for more information.


### Releasing

Access the [Ibis "Releasing" wiki](https://github.com/ibis-project/ibis/wiki/Releasing-Ibis) page
for more information.
