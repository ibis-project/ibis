# How to join an in-memory DataFrame to a TableExpression

You might have an in-memory DataFrame that you want to join to a TableExpression.
For example, you might have a file on your local machine that you don't want to upload to
your backend, but you need to join it to a table in that backend.

You can perform joins on local data to TableExpressions from your backend easily with Ibis MemTables.

In this guide, you will learn how to join a pandas DataFrame to a TableExpression.

## Data Setup:

In this example, we will create two DataFrames: one containing events and one containing event names.
We will save the events to a parquet file and read that as a TableExpression in the DuckDB backend.
We will then convert the event names DataFrame to a PandasInMemoryTable (MemTable), which is
a pandas DataFrame as a TableExpression and join the two expressions together as we would
two TableExpressions in a backend.

```python
    In [1]: import ibis

    In [2]: import pandas as pd
       ...: from datetime import date

    In [3]: # create a pandas DataFrame that we will convert to a
       ...: # PandasInMemoryTable (Ibis MemTable)
       ...: events = pd.DataFrame(
       ...:     {
       ...:         'event_id': range(4),
       ...:         'event_name': [f'e{k}' for k in range(4)],
       ...:     }
       ...: )

    In [4]: # Create a parquet file that we will read in using the DuckDB backend
       ...: # as a TableExpression
       ...: measures = pd.DataFrame({
       ...:     "event_id": [0] * 2 + [1] * 3 + [2] * 5 + [3] * 2
       ...:     ,"measured_on": map(
       ...:         date
       ...:         ,[2021] * 12, [6] * 4 + [5] * 6 + [7] * 2
       ...:         ,range(1, 13)
       ...:     )
       ...:     ,"measurement": None
       ...: })

    In [5]: measures.at[1, "measurement"] = 5.
       ...: measures.at[4, "measurement"] = 42.
       ...: measures.at[5, "measurement"] = 42.
       ...: measures.at[7, "measurement"] = 11.

    In [6]: # Save measures to parquet:
       ...: measures.to_parquet('measures.parquet')

    In [7]: # connect to a DuckDB backend
       ...: conn = ibis.connect('duckdb://:memory:')
       ...: measures = conn.register('measures.parquet', 'measures')

    In [8]: # `measures` is a TableExpression in a DuckDB backend connection:
       ...: measures
    Out[8]:
    AlchemyTable: measures
      event_id    int64
      measured_on date
      measurement float64
```

Converting a pandas DataFrame to a MemTable is as simple as feeding it to `ibis.memtable`:

```python
    In [9]: # To join, convert your DataFrame to a memtable
       ...: mem_events = ibis.memtable(events)

    In [10]: mem_events
    Out[10]:
    PandasInMemoryTable
      data:
        DataFrameProxy:
             event_id event_name
          0         0         e0
          1         1         e1
          2         2         e2
          3         3         e3
```

and joining is the same as joining any two TableExpressions:

```python
    In [11]: # Join as you would two table expressions
        ...: measures.join(
        ...:     mem_events,
        ...:     measures['event_id'] == mem_events['event_id']
        ...: ).execute()
    Out[11]:
        event_id measured_on  measurement  event_name
    0          0  2021-06-01          NaN          e0
    1          0  2021-06-02          5.0          e0
    2          1  2021-06-03          NaN          e1
    3          1  2021-06-04          NaN          e1
    4          1  2021-05-05         42.0          e1
    5          2  2021-05-06         42.0          e2
    6          2  2021-05-07          NaN          e2
    7          2  2021-05-08         11.0          e2
    8          2  2021-05-09          NaN          e2
    9          2  2021-05-10          NaN          e2
    10         3  2021-07-11          NaN          e3
    11         3  2021-07-12          NaN          e3
```

Note that the return result of the `join` is a TableExpression and that `execute` returns a pandas DataFrame.
