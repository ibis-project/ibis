#!/bin/bash

$PYTHON setup.py install --single-version-externally-managed --record=installed-files.txt
$PYTHON -c "import ibis; print(ibis.__version__.replace('v', ''))" > ibis/.version

# Add more build steps here, if they are necessary.

# See
# http://docs.continuum.io/conda/build.html
# for a list of environment variables that are set during the build process.
