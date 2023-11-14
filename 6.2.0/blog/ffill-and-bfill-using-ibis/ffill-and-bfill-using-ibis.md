# `ffill` and `bfill` using Ibis

**by Patrick Clarke**

Suppose you have a table of data mapping events and dates to values, and that this data contains gaps in values.

Suppose you want to forward fill these gaps such that, one-by-one,
if a value is null, it is replaced by the non-null value preceding.

For example, you might be measuring the total value of an account over time.
Saving the same value until that value changes is an inefficient use of space,
so you might only measure the value during certain events,
like a change in ownership or value.

In that case, to view the value of the account by day, you might want to interpolate dates
and then ffill or bfill value to show the account value over time by date.

Date interpolation will be covered in a different guide,
but if you already have the dates then you can fill in some values.

This was heavily inspired by Gil Forsyth's writeup on ffill and bfill on the
[Ibis GitHub Wiki](https://github.com/ibis-project/ibis/wiki/ffill-and-bfill-using-window-functions).

### Setup

First, we want to make some mock data.
To demonstrate this technique in a non-pandas backend, we will use the DuckDB backend.

Our data will have measurements by date, and these measurements will be grouped by an event id.
We will then save this data to `data.parquet` so we can register that parquet file as a table in our DuckDB connector.

```python
In [1]: import ibis; from datetime import date
In [2]: import numpy as np; import pandas as pd

In [3]: df = pd.DataFrame({
   ...:     "event_id": [0] * 2 + [1] * 3 + [2] * 5 + [3] * 2
   ...:     ,"measured_on": map(
   ...:         date
   ...:         ,[2021] * 12, [6] * 4 + [5] * 6 + [7] * 2
   ...:         ,range(1, 13)
   ...:     )
   ...:     ,"measurement": np.nan
   ...: })

In [4]: df.head()
Out[4]:
   event_id measured_on  measurement
0         0  2021-06-01          NaN
1         0  2021-06-02          NaN
2         1  2021-06-03          NaN
3         1  2021-06-04          NaN
4         1  2021-05-05          NaN

In [5]: df.at[1, "measurement"] = 5.
In [6]: df.at[4, "measurement"] = 42.
In [7]: df.at[5, "measurement"] = 42.
In [8]: df.at[7, "measurement"] = 11.

In [9]: df
Out[9]:
    event_id measured_on  measurement
0          0  2021-06-01          NaN
1          0  2021-06-02          5.0
2          1  2021-06-03          NaN
3          1  2021-06-04          NaN
4          1  2021-05-05         42.0
5          2  2021-05-06         42.0
6          2  2021-05-07          NaN
7          2  2021-05-08         11.0
8          2  2021-05-09          NaN
9          2  2021-05-10          NaN
10         3  2021-07-11          NaN
11         3  2021-07-12          NaN

In [10]: df.to_parquet("data.parquet")
```

To use the DuckDB backend with our data, we will spin up a DuckDB connection and then register `data.parquet` as `data`:

```python
In [11]: conn = ibis.connect('duckdb://:memory:')

In [12]: conn.register('data.parquet', table_name='data')
Out[12]:
AlchemyTable: data
  event_id    int64
  measured_on date
  measurement float64

In [13]: data = conn.table("data")

In [14]: data.execute()
Out[14]:
    event_id measured_on  measurement
0          0  2021-06-01          NaN
1          0  2021-06-02          5.0
2          1  2021-06-03          NaN
3          1  2021-06-04          NaN
4          1  2021-05-05         42.0
5          2  2021-05-06         42.0
6          2  2021-05-07          NaN
7          2  2021-05-08         11.0
8          2  2021-05-09          NaN
9          2  2021-05-10          NaN
10         3  2021-07-11          NaN
11         3  2021-07-12          NaN

In [15]: data
Out[15]:
AlchemyTable: data
  event_id    int64
  measured_on date
  measurement float64
```

### `ffill` Strategy

To better understand how we can forward-fill our gaps, let's take a minute to explain the strategy and then look at
the manual result.

We will partition our data by event groups and then sort those groups by date.

Our logic for forward fill is then: let `j` be an event group sorted by date and let `i` be a date within `j`.
If `i` is the first date in `j`, then continue.
If `i` is not the first date in `j`, then if `measurement` in `i` is null then replace it with `measurement` for `i-1`.
Otherwise, do nothing.

Let's take a look at what this means for the first few rows of our data:

```python
    event_id measured_on  measurement
0          0  2021-06-01          NaN # Since this is the first row of the event group (group 0), do nothing
1          0  2021-06-02          5.0 # Since this is not the first row of the group and is not null: do nothing
4          1  2021-05-05         42.0 # This is the first row of the event group (group 1): do nothing
2          1  2021-06-03          NaN # This is not the first row and is null: replace it (NaN → 42.0)
3          1  2021-06-04          NaN # This is not the first row and is null: replace it (NaN → 42.0)
5          2  2021-05-06         42.0 # This is the first row of the event group (group 2): do nothing
6          2  2021-05-07          NaN # This is not the first row and is null: replace it (NaN → 42.0)
7          2  2021-05-08         11.0 # This is not the first row and is not null: do nothing
8          2  2021-05-09          NaN # This is not the first row and is null: replace it (NaN → 11.0)
9          2  2021-05-10          NaN # This is not the first row and is null: replace it (NaN → 11.0)
10         3  2021-07-11          NaN # This is the first row of the event group (group 3): do nothing
11         3  2021-07-12          NaN # This is not the first row and is null: replace it (NaN → NaN)
```

Our result should for forward fill should look like this:

```python
    event_id measured_on  measurement
0          0  2021-06-01          NaN
1          0  2021-06-02          5.0
2          1  2021-05-05         42.0
3          1  2021-06-03         42.0
4          1  2021-06-04         42.0
5          2  2021-05-06         42.0
6          2  2021-05-07         42.0
7          2  2021-05-08         11.0
8          2  2021-05-09         11.0
9          2  2021-05-10         11.0
10         3  2021-07-11          NaN
11         3  2021-07-12          NaN
```

To accomplish this, we will create a window over our `event_id` to partition our data into groups.
We will take these groups and order them by `measured_on`:

```python
In [16]: win = ibis.window(group_by=data.event_id, order_by=data.measured_on, following=0)
```

Once we have our window defined, we can flag the first non-null value in an event group using `count`,
as it will count non-null values row-by-row within our group:

```python
In [17]: grouped = data.mutate(grouper=data.measurement.count().over(win))

In [18]: grouped.execute().sort_values(by=['event_id', 'measured_on'])
Out[18]:
    event_id measured_on  measurement  grouper
0          0  2021-06-01          NaN        0
1          0  2021-06-02          5.0        1
2          1  2021-05-05         42.0        1
3          1  2021-06-03          NaN        1
4          1  2021-06-04          NaN        1
5          2  2021-05-06         42.0        1
6          2  2021-05-07          NaN        1
7          2  2021-05-08         11.0        2
8          2  2021-05-09          NaN        2
9          2  2021-05-10          NaN        2
10         3  2021-07-11          NaN        0
11         3  2021-07-12          NaN        0
```

To see this a bit clearer: look at rows 0, 1, and 2.
Row 0 is NaN and is the first row of the group (event_id = 0), so at row 0 we have 0 non-null values (grouper = 0).
Row 1 is not null (5.0) and is the second row the group, so our count has increased by 1 (grouper = 1).
Row 2 is the first row of its group (event_id = 1) and is not null, so our count is 1 (grouper = 1).

Skip down to rows 9, 10, and 11.
Row 9 is the sixth row of group 2 and there are three non-null values in group 2 before row 9.
Therefore the count at row 9 is 3.

Row 10 is the first row of group 3 and is null, therefore its count is 0.
Finally: row 11 is the second row of group 3 and is null as well, therefore the count remains 0.

Under this design, we now have another partition.

Our first partition is by `event_id`.
Within each set in that partition, we have a partition by `grouper`, where each set has up to one non-null value.

Since there less than or equal to one non-null value in each group of `['event_id', 'grouper']`,
we can simply fill values by overwriting _all_ values within the group by the max value in the group.

So:

1. Group by `event_id` and `grouper`
2. Mutate the data along that grouping by populating a new column `ffill` with the `max` value of `measurement`.

```python
In [19]: result = (
    ...:     grouped
    ...:     .group_by([grouped.event_id, grouped.grouper])
    ...:     .mutate(ffill=grouped.measurement.max())
    ...:     .execute()
    ...: ).sort_values(by=['event_id', 'measured_on']).reset_index(drop=True)

In [20]: result
Out[20]:
    event_id measured_on  measurement  grouper  ffill
0          0  2021-06-01          NaN        0    NaN
1          0  2021-06-02          5.0        1    5.0
2          1  2021-05-05         42.0        1   42.0
3          1  2021-06-03          NaN        1   42.0
4          1  2021-06-04          NaN        1   42.0
5          2  2021-05-06         42.0        1   42.0
6          2  2021-05-07          NaN        1   42.0
7          2  2021-05-08         11.0        2   11.0
8          2  2021-05-09          NaN        2   11.0
9          2  2021-05-10          NaN        2   11.0
10         3  2021-07-11          NaN        0    NaN
11         3  2021-07-12          NaN        0    NaN
```

### `bfill` Strategy

Instead of sorting the dates ascending, we will sort them descending.
This is akin to starting at the last row in an event group and going backwards using the same logic outlined above.

Let's take a look:

```python
    event_id measured_on  measurement  grouper
0          0  2021-06-01          NaN        1 # null, take the previous row value (NaN → 5.0)
1          0  2021-06-02          5.0        1 # last row, do nothing
2          1  2021-05-05         42.0        1 # not null, do nothing
3          1  2021-06-03          NaN        0 # null, take previous row value (NaN → NaN)
4          1  2021-06-04          NaN        0 # last row, do nothing
5          2  2021-05-06         42.0        2 # not null, do nothing
6          2  2021-05-07          NaN        1 # null, take previous row value (NaN → 11.0)
7          2  2021-05-08         11.0        1 # not null, do nothing
8          2  2021-05-09          NaN        0 # null, take previous row value (NaN → NaN)
9          2  2021-05-10          NaN        0 # not null, do nothing
10         3  2021-07-11          NaN        0 # null, take previous row value (NaN → NaN)
11         3  2021-07-12          NaN        0 # last row, do nothing
```

Codewise, `bfill` follows the same strategy as `ffill`, we need to specify `order_by` to use `ibis.desc`.
This will flip our dates and our counts (therefore our `grouper`s) will start backwards.

```python
In [21]: win = ibis.window(group_by=data.event_id, order_by=ibis.desc(data.measured_on), following=0)

In [22]: grouped = data.mutate(grouper=data.measurement.count().over(win))

In [23]: grouped.execute().sort_values(by=['event_id', 'measured_on']).reset_index(drop=True)
Out[23]:
    event_id measured_on  measurement  grouper
0          0  2021-06-01          NaN        1
1          0  2021-06-02          5.0        1
2          1  2021-05-05         42.0        1
3          1  2021-06-03          NaN        0
4          1  2021-06-04          NaN        0
5          2  2021-05-06         42.0        2
6          2  2021-05-07          NaN        1
7          2  2021-05-08         11.0        1
8          2  2021-05-09          NaN        0
9          2  2021-05-10          NaN        0
10         3  2021-07-11          NaN        0
11         3  2021-07-12          NaN        0
```

And, again, if we take max of our `grouper` value, we will get the only non-null value if it exists:

```python
In [24]: result = (
    ...:     grouped
    ...:     .group_by([grouped.event_id, grouped.grouper])
    ...:     .mutate(bfill=grouped.measurement.max())
    ...:     .execute()
    ...: ).sort_values(by=['event_id', 'measured_on']).reset_index(drop=True)

In [25]: result
Out[25]:
    event_id measured_on  measurement  grouper  bfill
0          0  2021-06-01          NaN        1    5.0
1          0  2021-06-02          5.0        1    5.0
2          1  2021-05-05         42.0        1   42.0
3          1  2021-06-03          NaN        0    NaN
4          1  2021-06-04          NaN        0    NaN
5          2  2021-05-06         42.0        2   42.0
6          2  2021-05-07          NaN        1   11.0
7          2  2021-05-08         11.0        1   11.0
8          2  2021-05-09          NaN        0    NaN
9          2  2021-05-10          NaN        0    NaN
10         3  2021-07-11          NaN        0    NaN
11         3  2021-07-12          NaN        0    NaN
```

### `bfill` and `ffill` without Event Groups

You can `bfill` and `ffill` without event groups by ignoring that grouping.
Remove all references of `event_id` and you can treat the entire dataset as one event.

Your window function will increment whenever a new non-null value is observed, creating that partition where each
set has up to one non-null value.

For example, reasoning through `bfill`:

```python
In [26]: data.execute().sort_values(by=['measured_on'])
Out[26]:
    event_id measured_on  measurement
4          1  2021-05-05         42.0 # not null, do nothing
5          2  2021-05-06         42.0 # not null, do nothing
6          2  2021-05-07          NaN # null, take previous value (NaN → 11.0)
7          2  2021-05-08         11.0 # not null, do nothing
8          2  2021-05-09          NaN # null, take previous value (NaN → 5.0)
9          2  2021-05-10          NaN # null, take previous value (NaN → 5.0)
0          0  2021-06-01          NaN # null, take previous value (NaN → 5.0)
1          0  2021-06-02          5.0 # not null, do nothing
2          1  2021-06-03          NaN # null, take previous value (NaN → NaN)
3          1  2021-06-04          NaN # null, take previous value (NaN → NaN)
10         3  2021-07-11          NaN # null, take previous value (NaN → NaN)
11         3  2021-07-12          NaN # last row, do nothing

In [27]: win = ibis.window(order_by=ibis.desc(data.measured_on), following=0)

In [28]: grouped = data.mutate(grouper=data.measurement.count().over(win))

In [29]: result = (
    ...:     grouped
    ...:     .group_by([grouped.grouper])
    ...:     .mutate(bfill=grouped.measurement.max())
    ...: )

In [30]: result.execute().sort_values(by=['measured_on'])
Out[30]:
    event_id measured_on  measurement  grouper  bfill
9          1  2021-05-05         42.0        4   42.0
4          2  2021-05-06         42.0        3   42.0
11         2  2021-05-07          NaN        2   11.0
10         2  2021-05-08         11.0        2   11.0
8          2  2021-05-09          NaN        1    5.0
7          2  2021-05-10          NaN        1    5.0
6          0  2021-06-01          NaN        1    5.0
5          0  2021-06-02          5.0        1    5.0
3          1  2021-06-03          NaN        0    NaN
2          1  2021-06-04          NaN        0    NaN
1          3  2021-07-11          NaN        0    NaN
0          3  2021-07-12          NaN        0    NaN
```

As an exercise, try to take your time and reason your way through `ffill`.

Happy coding!
