# Forward and backward fill data using window functions

If you have gaps in your data and need to fill them in using a simple forward fill
(given an order, null values are replaced by the value preceding) or backward fill
(given an order, null values are replaced by the value following), then you can do this in Ibis:

=== "`ffill`"

    ~~~python
    # Create a window that orders your series, default ascending
    win = ibis.window(order_by=data.measured_on, following=0)
    # Create a grouping that is a rolling count of non-null values
    # This creates a partition where each set has no more than one non-null value
    grouped = data.mutate(grouper=data.measurement.count().over(win))
    # Group by your newly-created grouping and, in each set,
    # set all values to the one non-null value in that set (if it exists)
    result = (
        grouped
        .group_by([grouped.grouper])
        .mutate(ffill=grouped.measurement.max())
    )
    # execute to get a pandas dataframe, sort values in case your backend shuffles
    result.execute().sort_values(by=['measured_on'])
    ~~~

=== "`bfill`"

    ~~~python
    # Create a window that orders your series (use ibis.desc to get descending order)
    win = ibis.window(order_by=ibis.desc(data.measured_on), following=0)
    # Create a grouping that is a rolling count of non-null values
    # This creates a partition where each set has no more than one non-null value
    grouped = data.mutate(grouper=data.measurement.count().over(win))
    # Group by your newly-created grouping and, in each set,
    # set all values to the one non-null value in that set (if it exists)
    result = (
        grouped
        .group_by([grouped.grouper])
        .mutate(ffill=grouped.measurement.max())
    )
    # execute to get a pandas dataframe, sort values in case your backend shuffles
    result.execute().sort_values(by=['measured_on'])
    ~~~

If you have an event partition, which means there's another segment you need to consider
for your ffill or bfill operations, you can do this as well:

=== "`ffill` with event partition"

    ~~~python
    # Group your data by your event partition and then order your series (default ascending)
    win = ibis.window(group_by=data.event_id, order_by=data.measured_on, following=0)
    # Create a grouping that is a rolling count of non-null values within each event
    # This creates a partition where each set has no more than one non-null value
    grouped = data.mutate(grouper=data.measurement.count().over(win))
    # Group by your newly-created grouping and, in each set,
    # set all values to the one non-null value in that set (if it exists)
    result = (
        grouped
        .group_by([grouped.event_id, grouped.grouper])
        .mutate(ffill=grouped.measurement.max())
    )
    # execute to get a pandas dataframe, sort values in case your backend shuffles
    result.execute().sort_values(by=['event_id', 'measured_on'])
    ~~~

=== "`bfill` with event partition"

    ~~~python
    # Group your data by your event partition and then order your series (use ibis.desc for desc)
    win = ibis.window(group_by=data.event_id, order_by=ibis.desc(data.measured_on), following=0)
    # Create a grouping that is a rolling count of non-null values within each event
    # This creates a partition where each set has no more than one non-null value
    grouped = data.mutate(grouper=data.measurement.count().over(win))
    # Group by your newly-created grouping and, in each set,
    # set all values to the one non-null value in that set (if it exists)
    result = (
        grouped
        .group_by([grouped.event_id, grouped.grouper])
        .mutate(ffill=grouped.measurement.max())
    )
    # execute to get a pandas dataframe, sort values in case your backend shuffles
    result.execute().sort_values(by=['event_id', 'measured_on'])
    ~~~

We wrote a deeper dive into how this works on the ibis-project blog
[here](../blog/ffill-and-bfill-using-ibis.md).
