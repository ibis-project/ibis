#!/usr/bin/env python

# Copyright 2014 Cloudera Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

COMMS_EXT_ENABLED = False

requirements = []
extensions = []
cmdclass = {}

if COMMS_EXT_ENABLED:
    requirements.append('cython >= 0.21')

    from Cython.Distutils import build_ext
    from Cython.Build import cythonize
    import Cython

    if Cython.__version__ < '0.19.1':
        raise Exception('Please upgrade to Cython 0.19.1 or newer')

    cmdclass['build_ext'] = build_ext

from setuptools import setup
import os
import sys

from distutils.extension import Extension

MAJOR = 0
MINOR = 3
MICRO = 0
VERSION = '%d.%d.%d' % (MAJOR, MINOR, MICRO)

ISRELEASED = False

if not ISRELEASED:
    VERSION += '.dev'

    # todo: acquire git hash


from distutils.command.clean import clean as _clean
class clean(_clean):
    def run(self):
        _clean.run(self)
        for x in []:
            try:
                os.remove(x)
            except OSError:
                pass

cmdclass['clean'] = clean

with open('requirements.txt') as f:
    file_reqs = f.read().splitlines()
    requirements = requirements + file_reqs

PY26 = sys.version_info[0] == 2 and sys.version_info[1] == 6
if PY26:
  requirements.append('argparse')
  requirements.append('unittest2')

if COMMS_EXT_ENABLED:
    import numpy as np

    common_include = ['ibis/src', np.get_include()]
    comms_ext_libraries = []
    if sys.platform != 'darwin':
        # libuuid is available without additional linking as part of the base
        # BSD system on OS X, needs to be installed and linked on Linux,
        # though.
        comms_ext_libraries.append('uuid')

    comms_ext = Extension('ibis.comms',
                          ['ibis/comms.pyx',
                           'ibis/src/ipc_support.c'],
                          depends=['ibis/src/ipc_support.h'],
                          libraries=comms_ext_libraries,
                          include_dirs=common_include)
    extensions = cythonize([comms_ext])

setup(
    name='ibis-framework',
    packages=['ibis',
              'ibis.expr',
              'ibis.expr.tests',
              'ibis.sql',
              'ibis.sql.tests',
              'ibis.tests'],
    version=VERSION,
    package_data={'ibis': ['*.pxd', '*.pyx']},
    ext_modules=extensions,
    cmdclass=cmdclass,
    install_requires=requirements,
    description="Productivity-centric Python Big Data Framework",
    license='Apache License, Version 2.0',
    maintainer="Wes McKinney",
    maintainer_email="wes@cloudera.com"
)
