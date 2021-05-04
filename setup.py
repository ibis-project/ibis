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

impala_requires = ['hdfs>=2.0.16', 'sqlalchemy>=1.3', 'requests']
impala_requires.append('impyla[kerberos]>=0.15.0')

sqlite_requires = ['sqlalchemy>=1.3']
postgres_requires = sqlite_requires + ['psycopg2']
mysql_requires = sqlite_requires + ['pymysql']

kerberos_requires = ['requests-kerberos']
visualization_requires = ['graphviz']
clickhouse_requires = [
    'sqlalchemy<1.4',
    'clickhouse-sqlalchemy',
    'clickhouse-driver>=0.1.3',
    'clickhouse-cityhash',
]
hdf5_requires = ['tables>=3.0.0']

parquet_requires = ['pyarrow>=0.12.0']
spark_requires = ['pyspark>=2.4.3']

geospatial_requires = ['geoalchemy2', 'geopandas', 'shapely']

dask_requires = [
    'dask[dataframe, array]>=2.22.0',
]

all_requires = (
    impala_requires
    + postgres_requires
    + mysql_requires
    + kerberos_requires
    + visualization_requires
    + clickhouse_requires
    + hdf5_requires
    + parquet_requires
    + spark_requires
    + geospatial_requires
    + dask_requires
)

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
        'impala': impala_requires,
        'kerberos': kerberos_requires,
        'postgres': postgres_requires,
        'mysql': mysql_requires,
        'sqlite': sqlite_requires,
        'visualization': visualization_requires,
        'clickhouse': clickhouse_requires,
        'hdf5': hdf5_requires,
        'parquet': parquet_requires,
        'spark': spark_requires,
        'geospatial': geospatial_requires,
        'dask': dask_requires,
    },
    description="Productivity-centric Python Big Data Framework",
    long_description=LONG_DESCRIPTION,
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering',
    ],
    license='Apache License, Version 2.0',
    maintainer="Phillip Cloud",
    maintainer_email="phillip.cloud@twosigma.com",
    entry_points={
        'ibis.backends': [
            'pandas = ibis.backends.pandas',
            'csv = ibis.backends.csv',
            'parquet = ibis.backends.parquet',
            'hdf5 = ibis.backends.hdf5',
            'impala = ibis.backends.impala',
            'sqlite = ibis.backends.sqlite',
            'postgres = ibis.backends.postgres',
            'mysql = ibis.backends.mysql',
            'clickhouse = ibis.backends.clickhouse',
            'spark = ibis.backends.spark',
            'pyspark = ibis.backends.pyspark',
            'dask = ibis.backends.dask',
        ]
    },
)
