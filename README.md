# Ibis: Python data analysis framework for Hadoop and SQL engines

|Service|Status|
| -------------: | :---- |
| Documentation  | [![Documentation Status](https://img.shields.io/badge/docs-docs.ibis--project.org-blue.svg)](http://ibis-project.org) |
| Conda packages | [![Anaconda-Server Badge](https://anaconda.org/conda-forge/ibis-framework/badges/version.svg)](https://anaconda.org/conda-forge/ibis-framework) |
| PyPI           | [![PyPI](https://img.shields.io/pypi/v/ibis-framework.svg)](https://pypi.org/project/ibis-framework) |
| Ibis CI        | [![Build status](https://github.com/ibis-project/ibis/actions/workflows/ibis-main.yml/badge.svg)](https://github.com/ibis-project/ibis/actions/workflows/ibis-main.yml?query=branch%3Amaster) |
| Backend CI     | [![Build status](https://github.com/ibis-project/ibis/actions/workflows/ibis-backends.yml/badge.svg)](https://github.com/ibis-project/ibis/actions/workflows/ibis-backends.yml?query=branch%3Amaster) |
| Coverage       | [![Codecov branch](https://img.shields.io/codecov/c/github/ibis-project/ibis/master.svg)](https://codecov.io/gh/ibis-project/ibis) |


Ibis is a toolbox to bridge the gap between local Python environments, remote
storage, execution systems like Hadoop components (HDFS, Impala, Hive, Spark)
and SQL databases. Its goal is to simplify analytical workflows and make you
more productive.

Install Ibis from PyPI with:

```sh
pip install ibis-framework
```

or from conda-forge with

```sh
conda install ibis-framework -c conda-forge
```

Ibis currently provides tools for interacting with the following systems:

- [Apache Impala](https://impala.apache.org/)
- [Apache Kudu](https://kudu.apache.org/)
- [Hadoop Distributed File System (HDFS)](https://hadoop.apache.org/)
- [PostgreSQL](https://www.postgresql.org/)
- [MySQL](https://www.mysql.com/)
- [SQLite](https://www.sqlite.org/)
- [Pandas](https://pandas.pydata.org/) [DataFrames](http://pandas.pydata.org/pandas-docs/stable/dsintro.html#dataframe)
- [Clickhouse](https://clickhouse.yandex)
- [BigQuery](https://cloud.google.com/bigquery)
- [OmniSciDB](https://www.omnisci.com)
- [PySpark](https://spark.apache.org)
- [Dask](https://dask.org/) (Experimental)

Learn more about using the library at http://ibis-project.org.
