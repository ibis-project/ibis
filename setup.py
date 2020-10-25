#!/usr/bin/env python
"""Ibis setup module."""
import pathlib
import sys

from setuptools import find_packages, setup

import versioneer

LONG_DESCRIPTION = """
Ibis is a productivity-centric Python big data framework.

See http://ibis-project.org
"""

VERSION = sys.version_info.major, sys.version_info.minor

impala_requires = ['hdfs>=2.0.16', 'sqlalchemy>=1.1,<1.3.7', 'requests']
impala_requires.append('impyla[kerberos]>=0.15.0')

sqlite_requires = ['sqlalchemy>=1.1,<1.3.7']
postgres_requires = sqlite_requires + ['psycopg2']
mysql_requires = sqlite_requires + ['pymysql']

omniscidb_requires = ['pymapd==0.24', 'pyarrow']
kerberos_requires = ['requests-kerberos']
visualization_requires = ['graphviz']
clickhouse_requires = [
    'clickhouse-driver>=0.1.3',
    'clickhouse-cityhash',
]
bigquery_requires = [
    'google-cloud-bigquery[bqstorage,pandas]>=1.12.0,<2.0.0dev',
    'pydata-google-auth',
]
hdf5_requires = ['tables>=3.0.0']

parquet_requires = ['pyarrow>=0.12.0']
spark_requires = ['pyspark>=2.4.3']

geospatial_requires = ['geoalchemy2', 'geopandas', 'shapely']

all_requires = (
    impala_requires
    + postgres_requires
    + omniscidb_requires
    + mysql_requires
    + kerberos_requires
    + visualization_requires
    + clickhouse_requires
    + bigquery_requires
    + hdf5_requires
    + parquet_requires
    + spark_requires
    + geospatial_requires
)

develop_requires = all_requires + [
    'black',
    'click',
    'pydocstyle==4.0.1',
    'flake8',
    'isort',
    'mypy',
    'pre-commit',
    'pygit2',
    'pytest>=4.5',
]

install_requires = [
    line.strip()
    for line in pathlib.Path(__file__)
    .parent.joinpath('requirements.txt')
    .read_text()
    .splitlines()
]

setup(
    name='ibis-framework',
    url='https://github.com/ibis-project/ibis',
    packages=find_packages(),
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    install_requires=install_requires,
    python_requires='>=3.7',
    extras_require={
        'all': all_requires,
        'develop': develop_requires,
        'impala': impala_requires,
        'kerberos': kerberos_requires,
        'postgres': postgres_requires,
        'omniscidb': omniscidb_requires,
        'mysql': mysql_requires,
        'sqlite': sqlite_requires,
        'visualization': visualization_requires,
        'clickhouse': clickhouse_requires,
        'bigquery': bigquery_requires,
        'hdf5': hdf5_requires,
        'parquet': parquet_requires,
        'spark': spark_requires,
        'geospatial': geospatial_requires,
    },
    description="Productivity-centric Python Big Data Framework",
    long_description=LONG_DESCRIPTION,
    classifiers=[
        'Development Status :: 4 - Beta',
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering',
    ],
    license='Apache License, Version 2.0',
    maintainer="Phillip Cloud",
    maintainer_email="phillip.cloud@twosigma.com",
)
