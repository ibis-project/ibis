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
import ibis.backends.pandas
import pandas
import sql_to_ibis

df = pandas.DataFrame({"column1": [1, 2, 3], "column2": ["4", "5", "6"]})
ibis_table = ibis.backends.pandas.from_dataframe(
    df, name="my_table", client=ibis.backends.pandas.PandasClient({})
)
sql_to_ibis.register_temp_table(ibis_table, "my_table")
sql_to_ibis.query(
    "select column1, cast(column2 as integer) + 1 as my_col2 from my_table"
).execute()
```

This would output a dataframe that looks like:

| column1 | my_col2 |
| ------- | ------- |
| 1       | 5       |
| 2       | 6       |
| 3       | 7       |

## Ibis on [Fugue](https://github.com/fugue-project/fugue)

[Fugue](https://github.com/fugue-project/fugue) is a low-code abstraction layer letting users express
the workflows in SQL or Python end-to-end. The design philosophy of Fugue and Ibis is very aligned, and
Fugue is at a higher level of abstraction compared to Ibis. So the integration is very intuitive, Ibis
is also able to run on all the backends Fugue supports: Pandas, Spark, Dask and DuckDB. The value Fugue
adds to Ibis is the seamless integration of SQL semantics and scientific computing plus non-standard SQL
operations. The detailed tutorial can be found
[here](https://fugue-tutorials.readthedocs.io/tutorials/integrations/ibis.html)

Here is an example of a distributed inference pipeline:

```python
import pandas as pd
import fugue_ibis
from fugue import FugueWorkflow

# schema: *,pred:double
def predict(df: pd.DataFrame) -> pd.DataFrame:
    model = load_model("somefile")
    return df.assign(pred=model.predict(df))

def distributed_predict(file1, df2, dest):
    dag = FugueWorkflow()
    a = dag.load(file1).as_ibis()
    b = dag.df(df2).as_ibis()
    # ibis operations (you can do more here)
    joined = a.inner_join(b, a.key==b.key)[a, b.f2]
    filtered = joined[joined.f1>0]
    # back to fugue, apply predict distributedly and save
    filtered.as_fugue().transform(predict).save(dest)
    return dag

# test locally
distributed_predict(small_file, pandas_df2, temp_dest).run()

# run on spark when you have a SparkSession: session
distributed_predict(large_file, spark_df2, dest).run(session)
```
