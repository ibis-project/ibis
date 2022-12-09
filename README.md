# Ibis

[![Documentation Status](https://img.shields.io/badge/docs-docs.ibis--project.org-blue.svg)](http://ibis-project.org)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/ibis-framework/badges/version.svg)](https://anaconda.org/conda-forge/ibis-framework)
[![PyPI](https://img.shields.io/pypi/v/ibis-framework.svg)](https://pypi.org/project/ibis-framework)
[![Build status](https://github.com/ibis-project/ibis/actions/workflows/ibis-main.yml/badge.svg)](https://github.com/ibis-project/ibis/actions/workflows/ibis-main.yml?query=branch%3Amaster)
[![Build status](https://github.com/ibis-project/ibis/actions/workflows/ibis-backends.yml/badge.svg)](https://github.com/ibis-project/ibis/actions/workflows/ibis-backends.yml?query=branch%3Amaster)
[![Codecov branch](https://img.shields.io/codecov/c/github/ibis-project/ibis/master.svg)](https://codecov.io/gh/ibis-project/ibis)

## What is Ibis?

Ibis is a Python library that provides a lightweight, universal interface for data wrangling. It helps Python users explore and transform data of any size, stored anywhere.

Ibis has three primary components:

1. **A dataframe API for Python**.
   This means that Python users can write Ibis code to manipulate tabular data.
2. **Interfaces to 10+ query engines.**
   This means that wherever data is stored, data scientists can use Ibis as their API of choice to communicate with any of those query engines.
3. **Deferred execution**.
   Ibis uses deferred execution, meaning that execution of code is pushed to the query engine.
   This means users can execute at the speed of their backend, not their local computer.

## Why Use Ibis?

Ibis aims to be a future-proof solution to interacting with data using Python and can accomplish this goal through its main features:

- **Familiar API**: Ibis’s API design borrows from popular APIs like pandas and dplyr that most users already know and like to use.
- **Consistent syntax**: Ibis aims to be universal Python API for tabular data, big or small.
- **Deferred execution**: Ibis pushes code execution to the query engine and only moves required data into memory when it has to.
  This leads to more faster, more efficient analytics workflows
- **Interactive mode**: Ibis also provides an interactive mode, in which users can quickly diagnose problems, do exploratory data analysis, and mock up workflows locally.
- **10+ supported backends**: Ibis supports multiple query engines and DataFrame APIs.
  Use one interface to transform with your data wherever it lives: from DataFrames in pandas to parquet files through DuckDB to tables in BigQuery.
- **Minimize rewrites**: Depending on backend capabilities, teams can often keep most of their Ibis code the same whether a team changes anything on the backend, like increasing or decreasing computing power, changing the number or size of their databases, or switching backend engines.

## Common Use Cases

- **Speed up prototype to production.**
  Scale code written and tested locally to the cloud of distributed systems with minimum rewrites.
- **Boost performance of existing Python or pandas code.**
  For example a general rule of thumb for pandas is "Have 5 to 10 times as much RAM as the size of your dataset".
  When a dataset exceeds this rule, using in-memory frameworks, like pandas, can be slow.
  Instead, using Ibis will significantly speed up your workflows because of its deferred execution.
  Ibis also empowers you to switch to a faster database engine, without changing much of your code.
- **Get rid of long, error-prone, fstrings.**
  Ibis provides one syntax for multiple query engines and dataframe APIs that lets you avoid learning new flavors of SQL or other framework-specific code.
  Learn the syntax once and use that syntax anywhere.

## Backends

Ibis acts as a universal frontend to the following systems:

- [Apache Impala](https://ibis-project.org/docs/latest/backends/Impala/)
- [ClickHouse](https://ibis-project.org/docs/latest/backends/ClickHouse/)
- [Dask](https://ibis-project.org/docs/latest/backends/Dask/)
- [DuckDB](https://ibis-project.org/docs/latest/backends/DuckDB/)
- [Google BigQuery](https://ibis-project.org/docs/dev/backends/BigQuery/)
- [HeavyAI](https://github.com/heavyai/ibis-heavyai)
- [MySQL](https://ibis-project.org/docs/latest/backends/MySQL/)
- [Microsoft SQL Server](https://ibis-project.org/dev/latest/backends/MSSQL/)
- [Pandas](https://ibis-project.org/docs/latest/backends/Pandas/)
- [Polars](https://ibis-project.org/docs/dev/backends/Polars/)
- [PostgreSQL](https://ibis-project.org/docs/latest/backends/PostgreSQL/)
- [PySpark](https://ibis-project.org/docs/latest/backends/PySpark/)
- [Snowflake](https://ibis-project.org/docs/dev/backends/Snowflake) (experimental)
- [SQLite](https://ibis-project.org/docs/latest/backends/SQLite/)
- [Trino](https://ibis-project.org/docs/dev/backends/Trino/) (experimental)

The list of supported backends is continuously growing. Anyone can get involved
in adding new ones! Learn more about contributing to ibis in our contributing
docs at https://github.com/ibis-project/ibis/blob/master/docs/CONTRIBUTING.md

## Installation

Install Ibis from PyPI with:

```
pip install ibis-framework
```

Or from conda-forge with:

```
conda install ibis-framework -c conda-forge
```

(It’s a common mistake to `pip install ibis`. If you try to use Ibis and get errors early on try uninstalling `ibis` and installing `ibis-framework`)

For specific backends, include the backend name in brackets for PyPI:

```
pip install ibis-framework[duckdb]
```

Or use `ibis-$BACKEND` where `$BACKEND` the specific backend you want to use:

```
conda install ibis-postgres -c conda-forge
```

## Getting Started with Ibis

You can find a number of helpful tutorials on the Ibis website
[here](https://ibis-project.org/docs/latest/tutorial/01-Introduction-to-Ibis/)
including:

- [Introduction to Ibis](https://ibis-project.org/docs/latest/tutorial/01-Introduction-to-Ibis/)
- [Aggregating and Joining Data](https://ibis-project.org/docs/latest/tutorial/02-Aggregates-Joins/)
- [Creating and Inserting Data](https://ibis-project.org/docs/latest/tutorial/05-IO-Create-Insert-External-Data/)

You can also get started analyzing any dataset, anywhere with just a few lines of Ibis code.
Here’s an example of how to use Ibis with an SQLite database.

Download the SQLite database from the ibis-tutorial-data GCS (Google Cloud Storage) bucket, then connect to it using ibis.

```bash
# make a directory called geo_dir and add the geography database to that folder
mkdir -p geo_dir
curl -LsS -o geo_dir/geography.db 'https://storage.googleapis.com/ibis-tutorial-data/geography.db'
```

Connect to the the database and show the available tables

```python
>>> import ibis
>>> ibis.options.interactive = True
>>> connection = ibis.sqlite.connect('geo_dir/geography.db')
>>> connection.list_tables()
['countries', 'gdp', 'independence']
```

Choose the `countries` table and preview its first few rows

```python
>>> countries = connection.table('countries')
countries.head()
```

|     | iso_alpha2 | iso_alpha3 | iso_numeric | fips | name                 | capital          | area_km2 | population | continent |
| :-- | :--------- | :--------- | :---------- | :--- | :------------------- | :--------------- | :------- | :--------- | :-------- |
| 0   | AD         | AND        | 20          | AN   | Andorra              | Andorra la Vella | 468      | 84000      | EU        |
| 1   | AE         | ARE        | 784         | AE   | United Arab Emirates | Abu Dhabi        | 82880    | 4975593    | AS        |
| 2   | AF         | AFG        | 4           | AF   | Afghanistan          | Kabul            | 647500   | 29121286   | AS        |
| 3   | AG         | ATG        | 28          | AC   | Antigua and Barbuda  | St. Johns        | 443      | 86754      | NA        |
| 4   | AI         | AIA        | 660         | AV   | Anguilla             | The Valley       | 102      | 13254      | NA        |

```python
# Select the name, continent and population columns and filter them to only return countries from Asia

asian_countries = countries['name', 'continent', 'population'].filter(countries['continent'] == 'AS')
asian_countries.limit(6)
```

|     | name                 | continent | population |
| :-- | :------------------- | :-------- | :--------- |
| 0   | United Arab Emirates | AS        | 4975593    |
| 1   | Afghanistan          | AS        | 29121286   |
| 2   | Armenia              | AS        | 2968000    |
| 3   | Azerbaijan           | AS        | 8303512    |
| 4   | Bangladesh           | AS        | 156118464  |
| 5   | Bahrain              | AS        | 738004     |

## Community and Contributing

Ibis is an open source project and welcomes contributions from anyone in the community.
Read more about how you can contribute [here](https://github.com/ibis-project/ibis/blob/master/docs/CONTRIBUTING.md).
We care about keeping our community welcoming for all to participate and have a [code of conduct](https://github.com/ibis-project/ibis/blob/master/docs/CODE_OF_CONDUCT.md) to ensure this.
The Ibis project is open sourced under the [Apache License](https://github.com/ibis-project/ibis/blob/master/LICENSE.txt).

Join our community here:

- Twitter: https://twitter.com/IbisData
- Gitter: https://gitter.im/ibis-dev/Lobby
- StackOverflow: https://stackoverflow.com/questions/tagged/ibis

For more information visit our official website [here](https://ibis-project.org/docs/latest/).
