# Contributing to Ibis


## Clone the Repository

To contribute to ibis you need to clone the repository from GitHub:

    git clone https://github.com/ibis-project/ibis


## Set Up a Development Environment

1. [Install miniconda](https://docs.conda.io/en/latest/miniconda.html)
2. Create a Conda environment suitable for ibis development:
   
   If you are developing for the Impala, Kudu, HDFS, PostgreSQL, MySQL, SQLite, Pandas, Clickhouse, BigQuery, and/or OmniSciDB backend(s):

        conda env create -n ibis-dev --file ci/requirements-dev-3.7-main.yml

   If you are developing for the PySpark or Spark backend:

        conda env create -n ibis-dev --file ci/requirements-dev-3.7-pyspark-spark.yml

3. Activate the environment

        conda activate ibis-dev

4. Install your local copy of Ibis into the Conda environment. This also
   sets up a pre-commit hook to check style and formatting before committing.

        make develop


## Run the Test Suite

Contributor [Krisztián Szűcs](https://github.com/kszucs) has spent many hours
crafting an easy-to-use [docker-compose](https://docs.docker.com/compose/)
setup that enables ibis developers to get up and running quickly.

For those unfamiliar with ``docker``, and ``docker-compose``, here are some
rough steps on how to get things set up:

- Install ``docker-compose`` with ``pip install docker-compose``
- Install [docker](https://docs.docker.com/install/)
- Be sure to follow the [post-install instructions](https://docs.docker.com/install/linux/linux-postinstall/) if you are running on Linux.
- Log in to your Docker hub account with ``docker login`` (create an account at <https://hub.docker.com/> if you don't have one).

Here are the steps to start database services and run the test suite:

```sh
make --directory ibis init
make --directory ibis testall
```

Also you can run tests for a specific backend:

```sh
make --directory ibis testparallel BACKENDS='omniscidb impala'
```

or start database services for a specific backend:

```sh
make --directory ibis init BACKENDS='omniscidb impala'
```

*make for targets `test` and `testparallel` automatically do restart of services (as a prerequisite)*

You can also run ``pytest`` tests on the command line if you are not testing
integration with running database services. For example, to run all the tests
for the ``pandas`` backend:

```sh
pytest ./ibis/pandas
```


## Style and Formatting

We use [flake8](http://flake8.pycqa.org/en/latest/),
[black](https://github.com/psf/black) and
[isort](https://github.com/pre-commit/mirrors-isort) to ensure our code
is formatted and linted properly. If you have properly set up your development
environment by running ``make develop``, the pre-commit hooks should check
that your proposed changes continue to conform to our style guide.

We use [numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html) as
our standard format for docstrings.


## Commit Philosophy

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


## Commit/PR Messages

Well-structed commit messages allow us to generate comprehensive release notes
and make it very easy to understand what a commit/PR contributes to our
codebase. Commit messages and PR titles should be prefixed with a standard
code the states what kind of change it is. They fall broadly into 3 categories:
``FEAT (feature)``, ``BUG (bug)``, and ``SUPP (support)``. The ``SUPP``
category has some more fine-grained aliases that you can use, such as ``BLD``
(build), ``CI`` (continuous integration), ``DOC`` (documentation), ``TST``
(testing), and ``RLS`` (releases).


## Maintainer's Guide

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
