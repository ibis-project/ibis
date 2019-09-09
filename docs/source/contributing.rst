.. _contrib:

********************
Contributing to Ibis
********************

.. _contrib.running_tests:

Clone the Repository
--------------------
To contribute to ibis you need to clone the repository from GitHub:

.. code-block:: sh

   git clone https://github.com/ibis-project/ibis

Set Up a Development Environment
--------------------------------
#. `Install miniconda <https://docs.conda.io/en/latest/miniconda.html>`_
#. Create a Conda environment suitable for ibis development:

   .. code-block:: sh

      conda env create -n ibis-dev --file ci/requirements-3.7-dev.yml

#. Activate the environment

   .. code-block:: sh

      conda activate ibis-dev

#. Install your local copy of Ibis into the Conda environment. This also
   sets up a pre-commit hook to check style and formatting before committing.

   .. code-block:: sh

      make develop


Run the Test Suite
------------------

Contributor `Krisztián Szűcs <https://github.com/kszucs>`_ has spent many hours
crafting an easy-to-use `docker-compose <https://docs.docker.com/compose/>`_
setup that enables ibis developers to get up and running quickly.

For those unfamiliar with ``docker``, and ``docker-compose``, here are some
rough steps on how to get things set up:

    - Install ``docker-compose`` with ``pip install docker-compose``
    - Install `docker <https://docs.docker.com/install/>`_

      - Be sure to follow the
        `post-install instructions
        <https://docs.docker.com/install/linux/linux-postinstall/>`_
        if you are running on Linux.


Here are the steps to start database services and run the test suite:

.. code-block:: sh

   make --directory ibis init
   make --directory ibis testparallel


You can also run ``pytest`` tests on the command line if you are not testing
integration with running database services. For example, to run all the tests
for the ``pandas`` backend:

.. code-block:: sh

   pytest ./ibis/pandas


Style and Formatting
--------------------

We use `flake8 <http://flake8.pycqa.org/en/latest/>`_,
`black <https://github.com/psf/black>`_ and
`isort <https://github.com/pre-commit/mirrors-isort>`_ to ensure our code
is formatted and linted properly. If you have properly set up your development
environment by running ``make develop``, the pre-commit hooks should check
that your proposed changes continue to conform to our style guide.

We use `numpydoc <https://numpydoc.readthedocs.io/en/latest/format.html>`_ as
our standard format for docstrings.


Commit Philosophy
-----------------

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


Commit/PR Messages
------------------

Well-structed commit messages allow us to generate comprehensive release notes
and make it very easy to understand what a commit/PR contributes to our
codebase. Commit messages and PR titles should be prefixed with a standard
code the states what kind of change it is. They fall broadly into 3 categories:
``FEAT (feature)``, ``BUG (bug)``, and ``SUPP (support)``. The ``SUPP``
category has some more fine-grained aliases that you can use, such as ``BLD``
(build), ``CI`` (continuous integration), ``DOC`` (documentation), ``TST``
(testing), and ``RLS`` (releases).


Maintainer's Guide
------------------

Maintainers generally perform two roles, merging PRs and making official
releases.


Merging PRs
~~~~~~~~~~~

We have a CLI script that will merge Pull Requests automatically once they have
been reviewed and approved. See the help message in ``dev/merge-pr.py`` for
full details. If you have two-factor authentication turned on in Github, you
will have to generate an application-specific password by following this
`guide <https://help.github.com/en/articles/creating-a-personal-access-token-for-the-command-line>`_.
You will then use that generated password on the command line for the ``-P``
argument.


Releasing
~~~~~~~~~

TODO


***************
Code of Conduct
***************

Ibis is governed by the
`NumFOCUS code of conduct <https://numfocus.org/code-of-conduct>`_,
which in a short version is:

- Be kind to others. Do not insult or put down others. Behave professionally.
  Remember that harassment and sexist, racist, or exclusionary jokes are not
  appropriate for NumFOCUS.
- All communication should be appropriate for a professional audience
  including people of many different backgrounds. Sexual language and
  imagery is not appropriate.
- NumFOCUS is dedicated to providing a harassment-free community for everyone,
  regardless of gender, sexual orientation, gender identity, and expression,
  disability, physical appearance, body size, race, or religion. We do not
  tolerate harassment of community members in any form.
- Thank you for helping make this a welcoming, friendly community for all.
