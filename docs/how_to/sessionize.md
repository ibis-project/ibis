# How to Sessionize a Log of Events

Suppose you have entities (users, objects, actions, etc) that have event logs through polling or event triggers.

You might be interested in partitioning these logs by something called **sessions**, which can be defined as the duration of an event.

In the case of a user portal, it might be the time spent completing a task or navigating an app.
For games, it might be a time spent playing the game or remaining logged in.
For retail, it might be checking out or walking the premises.

This guide on sessionization is inspired by [_The Expressions API in Polars is Amazing_](https://www.pola.rs/posts/the-expressions-api-in-polars-is-amazing/),
a blog post in the [Polars](https://www.pola.rs/) community demonstrating the strength of polars expressions.

## Sessionizing Logs on a Cadence

For this example, we have a dataset that contains entities polled on a cadence.
The data used here can be found at `https://storage.googleapis.com/ibis-tutorial-data/wowah_data/wowah_data.csv`.
You can use `ibis.read("https://storage.googleapis.com/ibis-tutorial-data/wowah_data/wowah_data.csv")` to quickly get it into a table expression.

Our data contains the following:

- `char` : a unique identifier for a character (or a player). This is our entity column
- `timestamp`: a timestamp denoting when a `char` was polled. This occurs every ~10 minutes

We can take this information, along with a definition of what separates two sessions for an entity, and break our dataset up into sessions **without using any joins**:

```python
# Imports
import ibis
from ibis import _ as c

# Read files into table expressions with ibis.read:
data = ibis.read("https://storage.googleapis.com/ibis-tutorial-data/wowah_data/wowah_data_raw.parquet")

# integer delay in seconds noting if a row should be included in the previous session for an entity
session_boundary_threshold = 30 * 60

# Window for finding session ids per character
entity_window = ibis.cumulative_window(group_by=c.char, order_by=c.timestamp)

# Take the previous timestamp within a window (by character ordered by timestamp):
# Note: the first value in a window will be null
ts_lag = c.timestamp.lag().over(entity_window)

# Subtract the lag from the current timestamp to get a timedelta
ts_delta = c.timestamp - ts_lag

# Compare timedelta to our session delay in seconds to determine if the
# current timestamp falls outside of the session.
# Cast as int for aggregation
is_new_session = (ts_delta > ibis.interval(seconds=session_boundary_threshold))

# Window for finding session min/max
session_window = ibis.window(group_by=[c.char, c.session_id])

# Generate all of the data we need to analyze sessions:
sessionized = (
    data
    # Create a session id for each character by using a cumulative sum
    # over the `new_session` column
    .mutate(new_session=is_new_session.fillna(True))
    # Create a session id for each character by using a cumulative sum
    # over the `new_session` column
    .mutate(session_id=c.new_session.sum().over(entity_window))
    # Drop `new_session` because it is no longer needed
    .drop("new_session")
    .mutate(
        # Get session duration using max(timestamp) - min(timestamp) over our window
        session_duration=c.timestamp.max().over(session_window) - c.timestamp.min().over(session_window)
    )
    # Sort for convenience
    .order_by([c.char, c.timestamp])
)
```
