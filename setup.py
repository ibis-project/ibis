#!/usr/bin/env python

from setuptools import setup, find_packages

import versioneer


LONG_DESCRIPTION = """
Ibis is a productivity-centric Python big data framework.

See http://ibis-project.org
"""

impala_requires = [
    'hdfs>=2.0.16',
    'impyla>=0.14.0',
    'sqlalchemy>=1.0.0,<1.1.15',
]

sqlite_requires = ['sqlalchemy>=1.0.0,<1.1.15']
postgres_requires = sqlite_requires + ['psycopg2']
mysql_requires = sqlite_requires + ['pymysql']
kerberos_requires = ['requests-kerberos']
visualization_requires = ['graphviz']
clickhouse_requires = ['clickhouse-driver>=0.0.8']
bigquery_requires = ['google-cloud-bigquery<0.28']
hdf5_requires = ['tables>=3.0.0']
parquet_requires = ['pyarrow>=0.6.0']

all_requires = (
    impala_requires +
    postgres_requires +
    mysql_requires +
    kerberos_requires +
    visualization_requires +
    clickhouse_requires +
    bigquery_requires +
    hdf5_requires +
    parquet_requires
)

develop_requires = all_requires + [
    'click',
    'flake8',
    'pytest>=3',
]

with open('requirements.txt', 'rt') as f:
    install_requires = list(map(str.strip, f))

setup(
    name='ibis-framework',
    url='https://github.com/ibis-project/ibis',
    packages=find_packages(),
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    install_requires=install_requires,
    extras_require={
        'all': all_requires,
        'develop:python_version < "3"': develop_requires + [
            'thriftpy<=0.3.9',
            'thrift<=0.9.3',
            'mock',
        ],
        'develop:python_version >= "3"': develop_requires,
        'impala:python_version < "3"': impala_requires + [
            'thriftpy<=0.3.9',
            'thrift<=0.9.3',
        ],
        'impala:python_version >= "3"': impala_requires,
        'kerberos': kerberos_requires,
        'postgres': postgres_requires,
        'mysql': mysql_requires,
        'sqlite': sqlite_requires,
        'visualization': visualization_requires,
        'clickhouse:python_version != "3.4"': clickhouse_requires + [
            'clickhouse-cityhash'
        ],
        'clickhouse:python_version == "3.4"': clickhouse_requires,
        'bigquery': bigquery_requires,
        'csv:python_version < "3"': ['pathlib2'],
        'hdf5': hdf5_requires,
        'hdf5:python_version < "3"': hdf5_requires + ['pathlib2'],
        'parquet': parquet_requires,
        'parquet:python_version < "3"': parquet_requires + ['pathlib2'],
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
        'Programming Language :: Python :: 2.7',
        'Topic :: Scientific/Engineering',
    ],
    license='Apache License, Version 2.0',
    maintainer="Phillip Cloud",
    maintainer_email="phillip.cloud@twosigma.com"
)
