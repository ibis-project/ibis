# Ibis v3.1.0

**by Marlene Mhangami**

25 July 2022

## Introduction

Ibis 3.1 has officially been released as the latest version of the package.
With this release comes new convenience features, increased backend operation coverage and a plethora of bug fixes.
As usual, a full list of the changes can be found in the project release notes [here](../release_notes.md) Let’s talk about some of the new changes 3.1 brings for Ibis users.

## `ibis.connect`

The first significant change to note is that, Ibis now provides a more convenient way to connect to a backend using the `ibis.connect` method.
You can now use this function to connect to an appropriate backend using a connection string.

Here are some examples:

<!-- prettier-ignore-start -->

=== "DuckDB"

    Initialize a DuckDB instance using `'duckdb://:memory:'`
    ~~~python
    conn = ibis.connect('duckdb://:memory:')
    ~~~
    And begin registering your tables:
    ~~~python
    conn.register('csv://farm_data/dates.csv', 'dates')
    conn.register('csv://farm_data/farmer_groups.csv', 'farmer_groups')
    conn.register('csv://farm_data/crops.csv', 'crops')
    conn.register('csv://farm_data/farms.csv', 'farms')
    conn.register('csv://farm_data/harvest.csv', 'harvest')
    conn.register('csv://farm_data/farmers.csv', 'farmers')
    conn.register('csv://farm_data/tracts.csv', 'tracts')
    conn.register('csv://farm_data/fields.csv', 'fields')
    ~~~
    You can also do this programmatically:
    ~~~python
    files = glob.glob('farm_data/*.csv')

    for file in files:
        fname = 'csv://' + file
        tname = file.replace('farm_data/', '').replace('.csv', '')
        conn.register(fname, tname)
    ~~~
    This method isn’t limited to `csv://`.  It works with `parquet://` and `csv.gz://` as well.
    Give it a try!

=== "Postgres"

    ~~~python
    conn = ibis.connect('postgres://<username>:<password>@<host>:<port>/<database>')
    ~~~
    Or, using a [.pgpass file](https://www.postgresql.org/docs/9.3/libpq-pgpass.html):
    ~~~python
    conn = ibis.connect('postgres://<username>@<host>:<port>/<database>')
    ~~~
<!-- prettier-ignore-end -->

## Unnest Support

One of the trickier parts about working with data is that it doesn’t usually come organized in neat, predictable rows and columns.
Instead data often consists of rows that could contain a single bit of data or arrays of it.
When data is organized in layers, as with arrays, it can sometimes be difficult to work with.
Ibis 3.1 introduces the `unnest` function as a way to flatten arrays of data.

Unnest takes a column containing an array of values and separates the individual values into rows as shown:

Before Unnest:

```
    | col    |
    | ------ |
    | [1, 2] |
```

After Unnest:

```
    | col |
    | --- |
    |  1  |
    |  2  |
```

Here is a self-contained example of creating a dataset with an array and then unnesting it:

<!-- prettier-ignore-start -->

=== "DuckDB"

    ~~~python
    import ibis
    import pandas as pd

    # Parquet save path
    fname = 'array_data.parquet'

    # Mock Data
    data = [
        ['array_id', 'array_value']
        ,[1, [1, 3, 4]]
        ,[2, [2, 4, 5]]
        ,[3, [6, 8]]
        ,[4, [1, 6]]
    ]

    # Save as parquet
    pd.DataFrame(data[1:], columns=data[0]).to_parquet(fname)

    # Connect to the file using a DuckDB backend
    conn = ibis.connect(f"duckdb://{fname}")

    # Create a table expression for your loaded data
    array_data = conn.table("array_data")

    # Optionally execute the array data to preview
    array_data.execute()

    # select the unnested values with their corresponding IDs
    array_data.select(['array_id', array_data['array_value'].unnest()]).execute()
    ~~~

=== "Postgres"

    ~~~python
    import ibis
    import pandas as pd

    # Postgres connection string for user 'ibistutorials' with a valid .pgpass file in ~/
    # See https://www.postgresql.org/docs/9.3/libpq-pgpass.html for details on ~/.pgpass
    cstring = 'postgres://ibistutorials@localhost:5432/pg-ibis'

    # Mock Data
    data = [
        ['array_id', 'array_value']
        ,[1, [1, 3, 4]]
        ,[2, [2, 4, 5]]
        ,[3, [6, 8]]
        ,[4, [1, 6]]
    ]

    # Create a dataframe for easy loading
    df = pd.DataFrame(data[1:], columns=data[0])

    # Postgres backend connection
    conn = ibis.connect(cstring)

    # SQLAlchemy Types
    # Integer type
    int_type = ibis.backends.postgres.sa.types.INT()
    # Array type function
    arr_f = ibis.backends.postgres.sa.types.ARRAY

    # Load data to table using pd.DataFrame.to_sql
    df.to_sql(
        name='array_data'
        ,con=conn.con.connect()
        ,if_exists='replace'
        ,index=False
        ,dtype={
            'array_id': int_type
            ,'array_value': arr_f(int_type)
        }
    )

    # Array Data Table Expression
    array_data = conn.table("array_data")

    # Optionally execute to preview entire table
    # array_data.execute()

    # Unnest
    array_data.select(['array_id', array_data['array_value'].unnest()]).execute()
    ~~~

<!-- prettier-ignore-end -->

## `_` API

There is now a shorthand for lambda functions using underscore (`_`).
This is useful for chaining expressions to one another and helps reduce total line characters and appearances of lambdas.

For example, let’s use `array_data` from above.
We will unnest `array_value`, find the weighted average, and then sum in one expression:

```python
from ibis import _

(
    array_data
    .select([
        'array_id'
        # array_data returns a TableExpr, `_` here is shorthand
        # for that returned expression
        ,_['array_value'].unnest().name('arval')
        # we can use it instead of saying `array_data`
        ,(_['array_value'].length().cast('float')
          / _['array_value'].length().sum().cast('float')).name('wgt')
    ])
    # Since the above `select` statement returns a TableExpr, we can use
    # `_` to reference that one as well:
    .mutate(wgt_prod=_.arval * _.wgt)
    # And again:
    .aggregate(vsum=_.wgt_prod.sum(), vcount=_.wgt_prod.count())
    # And again:
    .mutate(wgt_mean=_.vsum / _.vcount)
).execute()
```

Note that if you import `_` directly from `ibis` (`from ibis import _`), the default `_`
object will lose its functionality, so be mindful if you have a habit of using it outside of Ibis.

## Additional Changes

Along with these changes, the operation matrix has had a few more holes filled.
Contributors should note that backend test data is now loaded dynamically.
Most users won’t be exposed to this update, but it should make contribution a bit more streamlined.

To see the full patch notes, go to the [patch notes page](../release_notes.md)

As always, Ibis is free and open source.
Contributions are welcome and encouraged–drop into the discussions, raise an issue, or put in a pull request.

Download ibis 3.1 today!
