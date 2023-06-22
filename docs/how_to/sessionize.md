# Sessionize a log of events

Suppose you have entities (users, objects, actions, etc) that have event logs through polling or event triggers.

You might be interested in partitioning these logs by something called **sessions**, which can be defined as groups of consecutive event records without long interruptions for a given entity.

In the case of a user portal, it might be grouping the navigation events that result in completing a task or buying a product.
For online games, it might be a the grouping of activity events of a given user playing the game while remaining logged in.

Sessionization can also be useful on longer time scales, for instance to reconstruct active subscription data from a raw payment or activity log, so as to model customer churn.

This guide on sessionization is inspired by [_The Expressions API in Polars is Amazing_](https://www.pola.rs/posts/the-expressions-api-in-polars-is-amazing/), a blog post in the [Polars](https://www.pola.rs/) community demonstrating the strength of Polars expressions.

## Sessionizing logs on a cadence

For this example, we use an activity log from the online game "World of Warcraft" with more than 10 million records for 37,354 unique players [made available](https://www.kaggle.com/datasets/mylesoneill/warcraft-avatar-history?select=wowah_data.csv) under the CC0 / Public Domain license. A copy of the data can be found at `https://storage.googleapis.com/ibis-tutorial-data/wowah_data/wowah_data_raw.parquet` (75 MB) under the parquet format to reduce load times. You can use `ibis.read_parquet` to quickly get it into a table expression via the default `DuckDB` backend.

This data contains the following fields:

- `char` : a unique identifier for a character (or a player). This is our entity column.
- `timestamp`: a timestamp denoting when a `char` was polled. This occurs every ~10 minutes.

We can take this information, along with a definition of what separates two sessions for an entity, and break our dataset up into sessions **without using any joins**:

```python
# Imports
import ibis
from ibis import deferred as c

# Read files into table expressions with ibis.read_parquet:
data = ibis.read_parquet(
    "https://storage.googleapis.com/ibis-tutorial-data/wowah_data/wowah_data_raw.parquet"
)

# Integer delay in seconds noting if a row should be included in the previous
# session for an entity.
session_boundary_threshold = 30 * 60

# Window for finding session ids per character
entity_window = ibis.cumulative_window(group_by=c.char, order_by=c.timestamp)

# Take the previous timestamp within a window (by character ordered by timestamp):
# Note: the first value in a window will be null.
ts_lag = c.timestamp.lag().over(entity_window)

# Subtract the lag from the current timestamp to get a timedelta.
ts_delta = c.timestamp - ts_lag

# Compare timedelta to our session delay in seconds to determine if the
# current timestamp falls outside of the session.
# Cast as int for aggregation.
is_new_session = (ts_delta > ibis.interval(seconds=session_boundary_threshold))

# Window to compute session min/max and duration.
session_window = ibis.window(group_by=[c.char, c.session_id])

# Generate all of the data we need to analyze sessions:
sessionized = (
    data
    # Create a session id for each character by using a cumulative sum
    # over the `new_session` column.
    .mutate(new_session=is_new_session.fillna(True))
    # Create a session id for each character by using a cumulative sum
    # over the `new_session` column.
    .mutate(session_id=c.new_session.sum().over(entity_window))
    # Drop `new_session` because it is no longer needed.
    .drop("new_session")
    .mutate(
        # Get session duration using max(timestamp) - min(timestamp) over our window.
        session_duration=c.timestamp.max().over(session_window) - c.timestamp.min().over(session_window)
    )
    # Sort for convenience.
    .order_by([c.char, c.timestamp])
)
```

Calling `ibis.show_sql(sessionized)` displays the SQL query and can be used to confirm that this Ibis table expression does not rely on any join operations.

Calling `sessionized.to_pandas()` should complete in less than a minute, depending on the speed of the internet connection to download the data and the number of CPU cores available to parallelize the processing of this nested query.
