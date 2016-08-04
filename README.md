[![circleci-badge](https://circleci.com/gh/cloudera/ibis.svg?style=shield&circle-token=b84ff8383cbb0d6788ee0f9635441cb962949a4f)](https://circleci.com/gh/cloudera/ibis/tree/master)

[![codecov.io](http://codecov.io/github/cloudera/ibis/coverage.svg?branch=master)](http://codecov.io/github/cloudera/ibis?branch=master)

Current release from Anaconda.org [![Anaconda-Server Badge](https://anaconda.org/conda-forge/ibis-framework/badges/version.svg)](https://anaconda.org/conda-forge/ibis-framework)

# Deepfield users
## How to make a change
First use a personal fork to make the changes that you need to make.
This allows you to make a pull request into both the cloudera offical repo and the deepfield repo.
Once you're pull request has been merged into the deepfield fork, make a new tag.
The tag schema is vx.x.x.x where the first three x's are the same as the cloudera release you're working off of.
For example, if you are working off of v0.8.1 then the new tag should be v0.8.1.0.
If this tag already exists, then your tag should be one higher, v0.8.1.1.
After making the tag, update the conda build and anaconda-deps.
Keep the anaconda version the same as the tag for consistency.


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
