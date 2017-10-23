[![CircleCI Status](https://circleci.com/gh/ibis-project/ibis.svg?style=shield&circle-token=b84ff8383cbb0d6788ee0f9635441cb962949a4f)](https://circleci.com/gh/ibis-project/ibis/tree/master)
[![AppVeyor Status](https://ci.appveyor.com/api/projects/status/github/ibis-project/ibis?branch=master&svg=true)](https://ci.appveyor.com/project/cpcloud/ibis-xh5g1)
[![Documentation Status](https://readthedocs.org/projects/ibis-project/badge/?version=latest)](http://ibis-project.readthedocs.io/en/latest/?badge=latest)

Current release from Anaconda.org [![Anaconda-Server Badge](https://anaconda.org/conda-forge/ibis-framework/badges/version.svg)](https://anaconda.org/conda-forge/ibis-framework)


# Ibis: Python data analysis framework for Hadoop and SQL engines

Ibis is a toolbox to bridge the gap between local Python environments and
remote storage and execution systems like Hadoop components (HDFS, Impala,
Hive, Spark) and SQL databases (Postgres, etc.). Its goal is to simplify
analytical workflows and make you more productive.

Install Ibis from PyPI with:

    $ pip install ibis-framework

At this time, Ibis provides tools for the interacting with the following
systems:

- [Apache Impala (incubating)](http://impala.io/)
- [Apache Kudu](http://getkudu.io)
- Hadoop Distributed File System (HDFS)
- PostgreSQL (Experimental)
- SQLite
- Direct execution of ibis expressions against pandas object (Experimental)
- [Clickhouse](https://clickhouse.yandex) (Experimental)

Learn more about using the library at http://docs.ibis-project.org and read the
project blog at http://ibis-project.org for news and updates.
