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

from setuptools import setup, find_packages
import sys

import versioneer

with open('requirements.txt') as f:
    requirements = f.read().splitlines()


LONG_DESCRIPTION = """
Ibis is a productivity-centric Python big data framework.

See http://ibis-project.org
"""

setup(
    name='ibis-framework',
    packages=find_packages(),
    version=versioneer.get_version(),
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
            'graphviz',
        ] + (['thriftpy<=0.3.9'] if sys.version_info.major >= 3 else []),
        'develop': [
            'flake8',
            'hdfs>=2.0.0',
            'impyla>=0.13.7',
            'psycopg2',
            'pytest>=3' if sys.version_info.major >= 3 else 'pytest<3',
            'sqlalchemy>=1.0.0',
            'thrift<=0.9.3',
            'graphviz',
        ] + (['thriftpy<=0.3.9'] if sys.version_info.major >= 3 else []) + (
            [] if sys.version_info.major >= 3 else ['mock']
        ),
        'impala': [
            'hdfs>=2.0.0',
            'impyla>=0.13.7',
            'sqlalchemy>=1.0.0',
            'thrift<=0.9.3',
        ] + (['thriftpy<=0.3.9'] if sys.version_info.major >= 3 else []),
        'kerberos': ['requests-kerberos'],
        'postgres': ['psycopg2', 'sqlalchemy>=1.0.0'],
        'sqlite': ['sqlalchemy>=1.0.0'],
        'visualization': ['graphviz'],
    },
    scripts=[
        os.path.join(
            os.path.dirname(__file__), 'scripts', 'test_data_admin.py'
        ),
    ],
    description="Productivity-centric Python Big Data Framework",
    long_description=LONG_DESCRIPTION,
    classifiers=[
        'Development Status :: 4 - Beta',
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 2.7',
        'Topic :: Scientific/Engineering',
    ],
    license='Apache License, Version 2.0',
    maintainer="Wes McKinney",
    maintainer_email="wes@cloudera.com"
)
