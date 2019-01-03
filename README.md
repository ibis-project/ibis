# Ibis: Python data analysis framework for Hadoop and SQL engines

|Service|Status|
| -------------: | :---- |
| Documentation  | [![Documentation Status](https://img.shields.io/badge/docs-docs.ibis--project.org-blue.svg)](http://docs.ibis-project.org) |
| Conda packages | [![Anaconda-Server Badge](https://anaconda.org/conda-forge/ibis-framework/badges/version.svg)](https://anaconda.org/conda-forge/ibis-framework) |
| PyPI           | ![PyPI](https://img.shields.io/pypi/v/ibis-framework.svg) |
| CircleCI       | [![CircleCI Status](https://circleci.com/gh/ibis-project/ibis.svg?style=shield&circle-token=b84ff8383cbb0d6788ee0f9635441cb962949a4f)](https://circleci.com/gh/ibis-project/ibis/tree/master) |
| Azure          | [![Azure Status](https://dev.azure.com/ibis-project/ibis/_apis/build/status/ibis-project.ibis)](https://dev.azure.com/ibis-project/ibis/_build) |
| Coverage       | ![Codecov branch](https://img.shields.io/codecov/c/github/ibis-project/ibis/master.svg) |


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
- [MySQL](https://www.mysql.com/) (Experimental)
- [SQLite](https://www.sqlite.org/)
- [Pandas](https://pandas.pydata.org/) [DataFrames](http://pandas.pydata.org/pandas-docs/stable/dsintro.html#dataframe) (Experimental)
- [Clickhouse](https://clickhouse.yandex)
- [BigQuery](https://cloud.google.com/bigquery)
- [MapD](https://www.mapd.com) (Experimental)

Learn more about using the library at http://docs.ibis-project.org.
