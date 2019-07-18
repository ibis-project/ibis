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
#. Create a conda environment suitable for ibis development:

   .. code-block:: sh

      conda env create -n ibis-dev --file ci/requirements-3.7-dev.yml

#. Activate the environment

   .. code-block:: sh

      conda activate ibis-dev

Run the Test Suite
------------------

Contributor `Krisztián Szűcs <https://github.com/kszucs>`_ has spent many hours
crafting an easy-to-use `docker-compose <https://docs.docker.com/compose/>`_
setup that enables ibis developers to get up and running quickly.

Here are the steps to start database services and run the test suite:

.. code-block:: sh

   make --directory ibis init
   make --directory ibis testparallel

Code of Conduct
---------------

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
