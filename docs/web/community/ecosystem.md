# Ecosystem

## [pandas](https://github.com/pandas-dev/pandas)

[pandas](https://pandas.pydata.org) is a Python package that provides fast, 
flexible, and expressive data structures designed to make working with "relational" or 
"labeled" data both easy and intuitive. It aims to be the fundamental high-level 
building block for doing practical, real world data analysis in Python. Additionally, 
it has the broader goal of becoming the most powerful and flexible open source data 
analysis / manipulation tool available in any language. It is already well on its way 
towards this goal.

## [SQLAlchemy](https://github.com/sqlalchemy/sqlalchemy)

[SQLAlchemy](https://www.sqlalchemy.org/) is the Python SQL toolkit and 
Object Relational Mapper that gives application developers the full power and 
flexibility of SQL. SQLAlchemy provides a full suite of well known enterprise-level 
persistence patterns, designed for efficient and high-performing database access, 
adapted into a simple and Pythonic domain language.

## [sql_to_ibis](https://github.com/zbrookle/sql_to_ibis)

[sql_to_ibis](https://github.com/zbrookle/sql_to_ibis) is a Python package that 
translates SQL syntax into ibis expressions. This allows users to use one unified SQL 
dialect to target many different backends, even those that don't traditionally
 support SQL.  

A good use case would be ease of migration between databases or backends. Suppose you
 were moving from SQLite to MySQL or from PostgresSQL to BigQuery. These
  frameworks all have very subtle differences in SQL dialects, but with sql_to_ibis, 
  these differences are automatically translated in Ibis.
  
Another good use case is pandas, which has no SQL support at all for querying a
 dataframe. With sql_to_ibis this is made possible.

For example,

```python
from ibis.pandas.api import from_dataframe, PandasClient
from pandas import read_csv
from sql_to_ibis import register_temp_table, query

df = read_csv("some_file.csv")
ibis_table = from_dataframe(df, name="my_table", client=PandasClient({}))
register_temp_table(ibis_table, "my_table")
query("select column1, column2 as my_col2 from my_table")
```