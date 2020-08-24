# Ecosystem

## [pandas](https://github.com/pandas-dev/pandas)

[pandas](https://github.com/pandas-dev/pandas) is a Python package that provides fast, 
flexible, and expressive data structures designed to make working with "relational" or 
"labeled" data both easy and intuitive. It aims to be the fundamental high-level 
building block for doing practical, real world data analysis in Python. Additionally, 
it has the broader goal of becoming the most powerful and flexible open source data 
analysis / manipulation tool available in any language. It is already well on its way 
towards this goal.

## [SQLAlchemy](https://github.com/sqlalchemy/sqlalchemy)

[SQLAlchemy](https://github.com/sqlalchemy/sqlalchemy) is the Python SQL toolkit and 
Object Relational Mapper that gives application developers the full power and 
flexibility of SQL. SQLAlchemy provides a full suite of well known enterprise-level 
persistence patterns, designed for efficient and high-performing database access, 
adapted into a simple and Pythonic domain language.

## [sql_to_ibis](https://github.com/zbrookle/sql_to_ibis)

[sql_to_ibis](https://github.com/zbrookle/sql_to_ibis) is a Python package that 
translates SQL syntax into ibis expressions. This allows users to use one unified SQL 
dialect to target many different backends.

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