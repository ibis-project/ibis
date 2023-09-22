# Ibis

[![Documentation Status](https://img.shields.io/badge/docs-docs.ibis--project.org-blue.svg)](http://ibis-project.org)
[![Project Chat](https://img.shields.io/badge/zulip-join_chat-purple.svg?logo=zulip)](https://ibis-project.zulipchat.com)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/ibis-framework/badges/version.svg)](https://anaconda.org/conda-forge/ibis-framework)
[![PyPI](https://img.shields.io/pypi/v/ibis-framework.svg)](https://pypi.org/project/ibis-framework)
[![Build status](https://github.com/ibis-project/ibis/actions/workflows/ibis-main.yml/badge.svg)](https://github.com/ibis-project/ibis/actions/workflows/ibis-main.yml?query=branch%3Amaster)
[![Build status](https://github.com/ibis-project/ibis/actions/workflows/ibis-backends.yml/badge.svg)](https://github.com/ibis-project/ibis/actions/workflows/ibis-backends.yml?query=branch%3Amaster)
[![Codecov branch](https://img.shields.io/codecov/c/github/ibis-project/ibis/master.svg)](https://codecov.io/gh/ibis-project/ibis)

## What is Ibis?

Ibis is a Python library that provides a lightweight, universal interface for data wrangling. It helps Python users explore and transform data of any size, stored anywhere.

Ibis has three primary components:

1. **A dataframe API for Python**.
   Python users can write Ibis code to manipulate tabular data.
2. **Interfaces to 15+ query engines.**
   Wherever data is stored, people can use Ibis as their API of choice to communicate with any of those query engines.
3. **Deferred execution**.
   Ibis uses deferred execution, so execution of code is pushed to the query engine.
   Users can execute at the speed of their backend, not their local computer.

## Why Use Ibis?

Ibis aims to be a future-proof solution to interacting with data using Python and can accomplish this goal through its main features:

- **Familiar API**: Ibis’s API design borrows from popular APIs like pandas and dplyr that most users already know and like to use.
- **Consistent syntax**: Ibis aims to be a universal Python API for tabular data of any size, big or small.
- **Deferred execution**: Ibis pushes code execution to the query engine and only moves required data into memory when necessary.
  Analytics workflows are faster and more efficient
- **Interactive mode**: Ibis provides an interactive mode in which users can quickly diagnose problems, explore data, and mock up workflows and pipelines locally.
- **10+ supported backends**: Ibis supports multiple query engines and DataFrame APIs.
  Use one interface to transform with your data wherever it lives: from DataFrames in pandas to Parquet files through DuckDB to tables in BigQuery.
- **Minimize rewrites**: Teams can often keep their Ibis code the same regardless of backend changes, like increasing or decreasing computing power, changing the number or size of their databases, or switching backends entirely.
- **Flexibility when you need it**: When Ibis doesn't support something, it provides a way to jump directly into SQL.

## Common Use Cases

- **Speed up prototype to production.**
  Scale code written and tested locally to a distributed system or cloud SQL engine with minimal rewrites.
- **Boost performance of existing Python or pandas code.**
  For example a general rule of thumb for pandas is "Have 5 to 10 times as much RAM as the size of your dataset".
  When a dataset exceeds this rule using in-memory frameworks like pandas can be slow.
  Instead, using Ibis will significantly speed up your workflows because of its deferred execution.
  Ibis also empowers you to switch to a faster database engine, without changing much of your code.
- **Get rid of long, error-prone, `f`-strings.**
  Ibis provides one syntax for multiple query engines and dataframe APIs that lets you avoid learning new flavors of SQL or other framework-specific code.
  Learn the syntax once and use that syntax anywhere.

## Backends

Ibis acts as a universal frontend to the following systems:

- [Apache Arrow DataFusion](https://ibis-project.org/backends/datafusion/) (experimental)
- [Apache Druid](https://ibis-project.org/backends/druid/) (experimental)
- [Apache Impala](https://ibis-project.org/backends/impala/)
- [Apache PySpark](https://ibis-project.org/backends/pyspark/)
- [BigQuery](https://ibis-project.org/backends/bigquery/)
- [ClickHouse](https://ibis-project.org/backends/clickhouse/)
- [Dask](https://ibis-project.org/backends/dask/)
- [DuckDB](https://ibis-project.org/backends/duckdb/)
- [HeavyAI](https://github.com/heavyai/ibis-heavyai)
- [MySQL](https://ibis-project.org/backends/mysql/)
- [Oracle](https://ibis-project.org/backends/oracle/) (experimental)
- [Pandas](https://ibis-project.org/backends/pandas/)
- [Polars](https://ibis-project.org/backends/polars/) (experimental)
- [PostgreSQL](https://ibis-project.org/backends/postgresql/)
- [SQL Server](https://ibis-project.org/backends/mssql/)
- [SQLite](https://ibis-project.org/backends/sqlite/)
- [Snowflake](https://ibis-project.org/backends/snowflake) (experimental)
- [Trino](https://ibis-project.org/backends/trino/) (experimental)

The list of supported backends is continuously growing. Anyone can get involved
in adding new ones! Learn more about contributing to ibis in our contributing
documentation at https://github.com/ibis-project/ibis/blob/master/docs/CONTRIBUTING.md

## Installation

Install Ibis from PyPI with:

```bash
pip install 'ibis-framework[duckdb]'
```

Or from conda-forge with:

```bash
conda install ibis-framework -c conda-forge
```

(It’s a common mistake to `pip install ibis`. If you try to use Ibis and get errors early on try uninstalling `ibis` and installing `ibis-framework`)

To discover ibis, we suggest starting with the DuckDB backend (which is included by default in the conda-forge package). The DuckDB backend is performant and fully featured.

To use ibis with other backends, include the backend name in brackets for PyPI:

```bash
pip install 'ibis-framework[postgres]'
```

Or use `ibis-$BACKEND` where `$BACKEND` is the specific backend you want to use when installing from conda-forge:

```bash
conda install ibis-postgres -c conda-forge
```

## Getting Started with Ibis

We provide a number of tutorial and example notebooks in the
[ibis-examples](https://github.com/ibis-project/ibis-examples). The easiest way
to try these out is through the online interactive notebook environment
provided here:
[![Binder](https://static.mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ibis-project/ibis-examples/main)

You can also get started analyzing any dataset, anywhere with just a few lines
of Ibis code. Here’s an example of how to use Ibis with a SQLite database.

Download the SQLite database from the `ibis-tutorial-data` GCS (Google Cloud
Storage) bucket, then connect to it using ibis.

```bash
curl -LsS -o geography.db 'https://storage.googleapis.com/ibis-tutorial-data/geography.db'
```

Connect to the database and show the available tables

```python
>>> import ibis
>>> from ibis import _
>>> ibis.options.interactive = True
>>> con = ibis.sqlite.connect("geography.db")
>>> con.tables
Tables
------
- countries
- gdp
- independence
```

Choose the `countries` table and preview its first few rows

```python
>>> countries = con.tables.countries
>>> countries.head()
┏━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━┓
┃ iso_alpha2 ┃ iso_alpha3 ┃ iso_numeric ┃ fips   ┃ name                 ┃ capital          ┃ area_km2 ┃ population ┃ continent ┃
┡━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━┩
│ string     │ string     │ int32       │ string │ string               │ string           │ float64  │ int32      │ string    │
├────────────┼────────────┼─────────────┼────────┼──────────────────────┼──────────────────┼──────────┼────────────┼───────────┤
│ AD         │ AND        │          20 │ AN     │ Andorra              │ Andorra la Vella │    468.0 │      84000 │ EU        │
│ AE         │ ARE        │         784 │ AE     │ United Arab Emirates │ Abu Dhabi        │  82880.0 │    4975593 │ AS        │
│ AF         │ AFG        │           4 │ AF     │ Afghanistan          │ Kabul            │ 647500.0 │   29121286 │ AS        │
│ AG         │ ATG        │          28 │ AC     │ Antigua and Barbuda  │ St. Johns        │    443.0 │      86754 │ NA        │
│ AI         │ AIA        │         660 │ AV     │ Anguilla             │ The Valley       │    102.0 │      13254 │ NA        │
└────────────┴────────────┴─────────────┴────────┴──────────────────────┴──────────────────┴──────────┴────────────┴───────────┘
```

Show the 5 least populous countries in Asia

```python

>>> (
...     countries.filter(_.continent == "AS")
...     .select("name", "population")
...     .order_by(_.population)
...     .limit(5)
... )
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ name                           ┃ population ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ string                         │ int32      │
├────────────────────────────────┼────────────┤
│ Cocos [Keeling] Islands        │        628 │
│ British Indian Ocean Territory │       4000 │
│ Brunei                         │     395027 │
│ Maldives                       │     395650 │
│ Macao                          │     449198 │
└────────────────────────────────┴────────────┘
```

## Community and Contributing

Ibis is an open source project and welcomes contributions from anyone in the community.

- Read [the contributing guide](https://github.com/ibis-project/ibis/blob/master/docs/CONTRIBUTING.md).
- We care about keeping the community welcoming for all. Check out [the code of conduct](https://github.com/ibis-project/ibis/blob/master/docs/CODE_OF_CONDUCT.md).
- The Ibis project is open sourced under the [Apache License](https://github.com/ibis-project/ibis/blob/master/LICENSE.txt).

Join our community by interacting on GitHub or chatting with us on [Zulip](https://ibis-project.zulipchat.com/).

For more information visit https://ibis-project.org/.
