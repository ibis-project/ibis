[![codecov.io](http://codecov.io/github/cloudera/ibis/coverage.svg?branch=master)](http://codecov.io/github/cloudera/ibis?branch=master)

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
- [Apache Kudu (incubating)](http://getkudu.io)
- Hadoop Distributed File System (HDFS)
- PostgreSQL (Experimental)
- SQLite

Learn more about using the library at http://docs.ibis-project.org and read the
project blog at http://ibis-project.org for news and updates.
