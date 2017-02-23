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

requirements = []
extensions = []
cmdclass = {}

from setuptools import setup, find_packages  # noqa
import os  # noqa
import sys  # noqa

import versioneer  # noqa

from distutils.extension import Extension  # noqa
from distutils.command.clean import clean as _clean  # noqa


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

PY2 = sys.version_info[0] == 2

if PY2:
    requirements.append('mock')


LONG_DESCRIPTION = """
Ibis is a productivity-centric Python big data framework.

See http://ibis-project.org
"""

setup(
    name='ibis-framework',
    packages=find_packages(),
    version=versioneer.get_version(),
    ext_modules=extensions,
    cmdclass=versioneer.get_cmdclass(),
    install_requires=requirements,
    extras_require={
        'all': [
            'hdfs>=2.0.0',
            'impyla>=0.13.7',
            'psycopg2',
            'sqlalchemy>=1.0.0',
            'thrift<=0.9.3',
            'thriftpy<=0.3.9',
        ] + (['thriftpy<=0.3.9'] if sys.version_info[0] >= 3 else []),
        'develop': [
            'hdfs>=2.0.0',
            'impyla>=0.13.7',
            'mock',
            'psycopg2',
            'pytest<=2.9.2',
            'sqlalchemy>=1.0.0',
            'thrift<=0.9.3',
        ] + (['thriftpy<=0.3.9'] if sys.version_info[0] >= 3 else []),
        'impala': [
            'hdfs>=2.0.0',
            'impyla>=0.13.7',
            'sqlalchemy>=1.0.0',
            'thrift<=0.9.3',
        ] + (['thriftpy<=0.3.9'] if sys.version_info[0] >= 3 else []),
        'kerberos': ['requests-kerberos'],
        'postgres': ['psycopg2', 'sqlalchemy>=1.0.0'],
        'sqlite': ['sqlalchemy>=1.0.0'],
    },
    description="Productivity-centric Python Big Data Framework",
    long_description=LONG_DESCRIPTION,
    classifiers=[
        'Development Status :: 4 - Beta',
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Cython',
        'Topic :: Scientific/Engineering',
    ],
    license='Apache License, Version 2.0',
    maintainer="Wes McKinney",
    maintainer_email="wes@cloudera.com"
)
