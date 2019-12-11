# Ibis: Python data analysis framework for Hadoop and SQL engines

|Service|Status|
| -------------: | :---- |
| Documentation  | [![Documentation Status](https://img.shields.io/badge/docs-docs.ibis--project.org-blue.svg)](http://docs.ibis-project.org) |
| Conda packages | [![Anaconda-Server Badge](https://anaconda.org/conda-forge/ibis-framework/badges/version.svg)](https://anaconda.org/conda-forge/ibis-framework) |
| PyPI           | [![PyPI](https://img.shields.io/pypi/v/ibis-framework.svg)](https://pypi.org/project/ibis-framework) |
| Azure          | [![Azure Status](https://dev.azure.com/ibis-project/ibis/_apis/build/status/ibis-project.ibis)](https://dev.azure.com/ibis-project/ibis/_build) |
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
- [MSSQL](https://www.microsoft.com/en-us/sql-server/)
- [MySQL](https://www.mysql.com/) (Experimental)
- [SQLite](https://www.sqlite.org/)
- [Pandas](https://pandas.pydata.org/) [DataFrames](http://pandas.pydata.org/pandas-docs/stable/dsintro.html#dataframe) (Experimental)
- [Clickhouse](https://clickhouse.yandex)
- [BigQuery](https://cloud.google.com/bigquery)
- [OmniSciDB](https://www.omnisci.com) (Experimental)
- [Spark](https://spark.apache.org) (Experimental)

Learn more about using the library at http://docs.ibis-project.org.


## Notes

- OmniSciDB backend support is tested against a development release
of their database using the ``omnisci/core-os-cpu-dev`` Docker image.
Check the docker image tag used at
[docker-compose.yml](https://github.com/ibis-project/ibis/blob/master/ci/docker-compose.yml).
Some features may not work on earlier releases.
