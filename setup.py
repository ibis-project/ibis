# -*- coding: utf-8 -*-
from setuptools import setup

packages = [
    "ibis",
    "ibis.backends",
    "ibis.backends.base",
    "ibis.backends.base.file",
    "ibis.backends.base.sql",
    "ibis.backends.base.sql.alchemy",
    "ibis.backends.base.sql.compiler",
    "ibis.backends.base.sql.registry",
    "ibis.backends.clickhouse",
    "ibis.backends.clickhouse.tests",
    "ibis.backends.dask",
    "ibis.backends.dask.execution",
    "ibis.backends.dask.tests",
    "ibis.backends.dask.tests.execution",
    "ibis.backends.datafusion",
    "ibis.backends.datafusion.tests",
    "ibis.backends.impala",
    "ibis.backends.impala.tests",
    "ibis.backends.mysql",
    "ibis.backends.mysql.tests",
    "ibis.backends.pandas",
    "ibis.backends.pandas.execution",
    "ibis.backends.pandas.tests",
    "ibis.backends.pandas.tests.execution",
    "ibis.backends.postgres",
    "ibis.backends.postgres.tests",
    "ibis.backends.pyspark",
    "ibis.backends.pyspark.tests",
    "ibis.backends.sqlite",
    "ibis.backends.sqlite.tests",
    "ibis.backends.tests",
    "ibis.common",
    "ibis.expr",
    "ibis.expr.operations",
    "ibis.expr.types",
    "ibis.tests",
    "ibis.tests.benchmarks",
    "ibis.tests.expr",
    "ibis.tests.sql",
    "ibis.udf",
]

package_data = {"": ["*"]}

install_requires = [
    "atpublic>=2.3,<3",
    "cached_property>=1,<2",
    "multipledispatch>=0.6,<0.7",
    "numpy>=1,<2",
    "pandas>=1.2.5,<2",
    "parsy>=1.3.0,<2",
    "pydantic>=1.9.0,<2",
    "regex>=2021.7.6",
    "toolz>=0.11,<0.12",
]

extras_require = {
    ':python_version < "3.8"': [
        "importlib-metadata>=4,<5",
        "typing-extensions>=4.0.1,<5.0.0",
    ],
    "all": [
        "clickhouse-driver>=0.1,<0.3",
        "dask[array,dataframe]>=2021.10.0",
        "datafusion>=0.4,<0.5",
        "geoalchemy2>=0.6,<0.11",
        "geopandas>=0.6,<0.11",
        "graphviz>=0.16,<0.20",
        "hdfs[kerberos]>=2,<3",
        "impyla[kerberos]>=0.17,<0.19",
        "psycopg2>=2.7.6,<3",
        "pyarrow>=1,<8",
        "pymysql>=1,<2",
        "pyspark>=2.4.3,<4",
        "requests>=2,<3",
        "shapely>=1.6,<2",
        "sqlalchemy>=1.3,<1.4",
    ],
    "clickhouse": ["clickhouse-driver>=0.1,<0.3"],
    "dask": ["dask[array,dataframe]>=2021.10.0", "pyarrow>=1,<8"],
    "datafusion": ["datafusion>=0.4,<0.5"],
    "geospatial": [
        "geoalchemy2>=0.6,<0.11",
        "geopandas>=0.6,<0.11",
        "shapely>=1.6,<2",
    ],
    "impala": [
        "hdfs[kerberos]>=2,<3",
        "impyla[kerberos]>=0.17,<0.19",
        "requests>=2,<3",
    ],
    "mysql": ["pymysql>=1,<2", "sqlalchemy>=1.3,<1.4"],
    "postgres": ["psycopg2>=2.7.6,<3", "sqlalchemy>=1.3,<1.4"],
    "pyspark": ["pyarrow>=1,<8", "pyspark>=2.4.3,<4"],
    "sqlite": ["sqlalchemy>=1.3,<1.4"],
    "visualization": ["graphviz>=0.16,<0.20"],
}

entry_points = {
    "ibis.backends": [
        "clickhouse = ibis.backends.clickhouse",
        "dask = ibis.backends.dask",
        "datafusion = ibis.backends.datafusion",
        "impala = ibis.backends.impala",
        "mysql = ibis.backends.mysql",
        "pandas = ibis.backends.pandas",
        "postgres = ibis.backends.postgres",
        "pyspark = ibis.backends.pyspark",
        "spark = ibis.backends.pyspark",
        "sqlite = ibis.backends.sqlite",
    ]
}

setup_kwargs = {
    "name": "ibis-framework",
    "version": "2.1.1",
    "description": "Productivity-centric Python Big Data Framework",
    "long_description": "# Ibis: Python data analysis framework for Hadoop and SQL engines\n\n|        Service | Status                                                                                                                                                                                                |\n| -------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |\n|  Documentation | [![Documentation Status](https://img.shields.io/badge/docs-docs.ibis--project.org-blue.svg)](http://ibis-project.org)                                                                                 |\n| Conda packages | [![Anaconda-Server Badge](https://anaconda.org/conda-forge/ibis-framework/badges/version.svg)](https://anaconda.org/conda-forge/ibis-framework)                                                       |\n|           PyPI | [![PyPI](https://img.shields.io/pypi/v/ibis-framework.svg)](https://pypi.org/project/ibis-framework)                                                                                                  |\n|        Ibis CI | [![Build status](https://github.com/ibis-project/ibis/actions/workflows/ibis-main.yml/badge.svg)](https://github.com/ibis-project/ibis/actions/workflows/ibis-main.yml?query=branch%3Amaster)         |\n|     Backend CI | [![Build status](https://github.com/ibis-project/ibis/actions/workflows/ibis-backends.yml/badge.svg)](https://github.com/ibis-project/ibis/actions/workflows/ibis-backends.yml?query=branch%3Amaster) |\n|       Coverage | [![Codecov branch](https://img.shields.io/codecov/c/github/ibis-project/ibis/master.svg)](https://codecov.io/gh/ibis-project/ibis)                                                                    |\n\nIbis is a toolbox to bridge the gap between local Python environments, remote\nstorage, execution systems like Hadoop components (HDFS, Impala, Hive, Spark)\nand SQL databases. Its goal is to simplify analytical workflows and make you\nmore productive.\n\nInstall Ibis from PyPI with:\n\n```sh\npip install ibis-framework\n```\n\nor from conda-forge with\n\n```sh\nconda install ibis-framework -c conda-forge\n```\n\nIbis currently provides tools for interacting with the following systems:\n\n- [Apache Impala](https://impala.apache.org/)\n- [Apache Kudu](https://kudu.apache.org/)\n- [Hadoop Distributed File System (HDFS)](https://hadoop.apache.org/)\n- [PostgreSQL](https://www.postgresql.org/)\n- [MySQL](https://www.mysql.com/)\n- [SQLite](https://www.sqlite.org/)\n- [Pandas](https://pandas.pydata.org/) [DataFrames](http://pandas.pydata.org/pandas-docs/stable/dsintro.html#dataframe)\n- [Clickhouse](https://clickhouse.yandex)\n- [BigQuery](https://cloud.google.com/bigquery)\n- [OmniSciDB](https://www.omnisci.com)\n- [PySpark](https://spark.apache.org)\n- [Dask](https://dask.org/) (Experimental)\n\nLearn more about using the library at http://ibis-project.org.\n",
    "author": "Ibis Contributors",
    "author_email": None,
    "maintainer": "Ibis Contributors",
    "maintainer_email": None,
    "url": "https://ibis-project.org",
    "packages": packages,
    "package_data": package_data,
    "install_requires": install_requires,
    "extras_require": extras_require,
    "entry_points": entry_points,
    "python_requires": ">=3.7.1,<3.11",
}


setup(**setup_kwargs)
