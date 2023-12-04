from __future__ import annotations

import numpy as np
import pandas as pd

import ibis.expr.operations as ops
from ibis.backends.pandas.newpandas import execute
from ibis.backends.pandas.newutils import asseries


class UngroupedFrame:
    def __init__(self, df):
        self.df = df

    def groups(self):
        yield self.df

    def apply(self, func, **kwargs):
        return func(self.df, **kwargs)


class GroupedFrame:
    def __init__(self, df, group_keys):
        self.df = df
        self.group_keys = group_keys
        self.groupby = df.groupby(group_keys, as_index=True)

    def groups(self):
        for _, df in self.groupby:
            yield df

    def apply(self, func, **kwargs):
        result = self.groupby.apply(func, **kwargs)
        if len(self.group_keys) == 1:
            key, = self.group_keys
            return self.df[key].map(result)
        else:
            raise NotImplementedError("Only single group key is supported")



class RowsFrame:
    def __init__(self, parent, prepend_nan, append_nan):
        self.parent = parent
        self.prepend_nan = prepend_nan
        self.append_nan = append_nan

    def apply(self, func, **kwargs):
        results = {}
        for df in self.parent.groups():
            for i, (ix, row) in enumerate(df.iterrows()):
                start = row['_start']
                end = row['_end']

                if start is None and end is None:
                    subdf = df
                elif start is None:
                    subdf = df.iloc[:i + end + 1]

                elif end is None:
                    subdf = df.iloc[i + start:]
                else:
                    subdf = df.iloc[i + start:i + end + 1]

                res = func(subdf, **kwargs)
                if isinstance(res, pd.Series):
                    results[ix] = res[ix]
                else:
                    results[ix] = res

        return pd.Series(results)



@execute.register(ops.WindowBoundary)
def execute_window_boundary(op, value, preceding):
    return value


@execute.register(ops.WindowFrame)
def execute_window_frame(op, table, start, end, group_by, order_by, **kwargs):
    if start is not None:
        start = asseries(start, len(table))
        if op.start.preceding:
            start = -start
    if end is not None:
        end = asseries(end, len(table))
        if op.end.preceding:
            end = -end

    table = table.assign(
        _start=start,
        _end=end,
    )

    group_keys = [group.name for group in group_by]
    order_keys = [key.name for key in order_by]
    ascending = [key.ascending for key in op.order_by]

    if order_by:
        table = table.sort_values(order_keys, ascending=ascending, kind="mergesort")

    if group_by:
        frame = GroupedFrame(df=table, group_keys=group_keys)
    else:
        frame = UngroupedFrame(df=table)

    if start is None and end is None:
        return frame
    elif op.how == "rows":
        return RowsFrame(
            parent=frame,
            prepend_nan=False,
            append_nan=False,
        )
    else:
        raise NotImplementedError("Only rows window frame is supported")


@execute.register(ops.WindowFunction)
def execute_window_function(op, func, frame):
    return frame.apply(func)


@execute.register(ops.Lag)
def execute_lag(op, arg, offset, default):
    def agg(df):
        return df[arg.name].shift(1, fill_value=default)
    return agg


@execute.register(ops.Lead)
def execute_lead(op, arg, offset, default):
    def agg(df):
        return df[arg.name].shift(-1, fill_value=default)
    return agg
