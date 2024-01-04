from __future__ import annotations

import itertools
from typing import Callable

import numpy as np
import pandas as pd

from ibis.util import gen_name


def asseries(value, size=1):
    """Ensure that value is a pandas Series object, broadcast if necessary."""
    if isinstance(value, pd.Series):
        return value
    elif isinstance(value, (list, np.ndarray)):
        return pd.Series(itertools.repeat(np.array(value), size))
    else:
        return pd.Series(np.repeat(value, size))


def asframe(values: dict | tuple, concat=True):
    """Construct a DataFrame from a dict or tuple of Series objects."""
    if isinstance(values, dict):
        names, values = zip(*values.items())
    elif isinstance(values, tuple):
        names = [f"_{i}" for i in range(len(values))]
    else:
        raise TypeError(f"values must be a dict, or tuple; got {type(values)}")

    size = 1
    all_scalars = True
    for v in values:
        if isinstance(v, pd.Series):
            size = len(v)
            all_scalars = False
            break

    columns = [asseries(v, size) for v in values]
    if concat:
        df = pd.concat(columns, axis=1, keys=names).reset_index(drop=True)
        return df, all_scalars
    else:
        return columns, all_scalars


def generic(func: Callable, operands):
    return func(*operands.values())


def rowwise(func: Callable, operands):
    # dealing with a collection of series objects
    df, all_scalars = asframe(operands)
    result = df.apply(func, axis=1)  # , **kwargs)
    return result.iat[0] if all_scalars else result


def columnwise(func: Callable, operands):
    df, all_scalars = asframe(operands)
    result = func(df)
    return result.iat[0] if all_scalars else result


def serieswise(func, operands):
    (key, value), *rest = operands.items()
    if isinstance(value, pd.Series):
        # dealing with a single series object
        return func(**operands)
    else:
        # dealing with a single scalar object
        value = pd.Series([value])
        operands = {key: value, **dict(rest)}
        return func(**operands).iat[0]


def elementwise(func, operands):
    value = operands.pop(next(iter(operands)))
    if isinstance(value, pd.Series):
        # dealing with a single series object
        if operands:
            return value.apply(func, **operands)
        else:
            return value.map(func, na_action="ignore")
    else:
        # dealing with a single scalar object
        return func(value, **operands)


def agg(func, arg_column, where_column):
    if where_column is None:

        def applier(df):
            return func(df[arg_column.name])
    else:

        def applier(df):
            mask = df[where_column.name]
            col = df[arg_column.name][mask]
            return func(col)

    return applier


class UngroupedFrame:
    def __init__(self, df):
        self.df = df

    def groups(self):
        yield self.df

    def apply_reduction(self, func, **kwargs):
        result = func(self.df, **kwargs)
        data = [result] * len(self.df)
        return pd.Series(data, index=self.df.index)

    def apply_analytic(self, func, **kwargs):
        return func(self.df, **kwargs)


class GroupedFrame:
    def __init__(self, df, group_keys):
        self.df = df
        self.group_keys = group_keys
        self.groupby = df.groupby(group_keys, as_index=True)

    def groups(self):
        for _, df in self.groupby:
            yield df

    def apply_analytic(self, func, **kwargs):
        results = [func(df, **kwargs) for df in self.groups()]
        return pd.concat(results)

    def apply_reduction(self, func, **kwargs):
        name = gen_name("result")
        result = self.groupby.apply(func, **kwargs).rename(name)
        df = self.df.merge(result, left_on=self.group_keys, right_index=True)
        return df[name]


class RowsFrame:
    def __init__(self, parent):
        self.parent = parent

    @staticmethod
    def adjust(length, index, start_offset, end_offset):
        if start_offset is None:
            start_index = 0
        else:
            start_index = index + start_offset
            if start_index < 0:
                start_index = 0
            elif start_index > length:
                start_index = length

        if end_offset is None:
            end_index = length
        else:
            end_index = index + end_offset + 1
            if end_index < 0:
                end_index = 0
            elif end_index > length:
                end_index = length

        return (start_index, end_index)

    def apply_analytic(self, func, **kwargs):
        return self.parent.apply_analytic(func, **kwargs)

    def apply_reduction(self, func, **kwargs):
        results = {}
        for df in self.parent.groups():
            for i, (ix, row) in enumerate(df.iterrows()):
                # TODO(kszucs): use unique column names for _start, _end
                start, end = row["__start__"], row["__end__"]
                start_index, end_index = self.adjust(len(df), i, start, end)
                subdf = df.iloc[start_index:end_index]
                results[ix] = func(subdf, **kwargs)

        return pd.Series(results)


class RangeFrame:
    def __init__(self, parent, order_key):
        self.parent = parent
        self.order_key = order_key

    @staticmethod
    def predicate(col, i, start, end):
        value = col.iat[i]
        if start is None:
            return col <= value + end
        elif end is None:
            return col >= value + start
        else:
            return (col >= value + start) & (col <= value + end)

    def apply_analytic(self, func, **kwargs):
        return self.parent.apply_analytic(func, **kwargs)

    def apply_reduction(self, func, **kwargs):
        results = {}
        for df in self.parent.groups():
            for i, (ix, row) in enumerate(df.iterrows()):
                start, end = row["__start__"], row["__end__"]
                column = df[self.order_key]
                predicate = self.predicate(column, i, start, end)
                subdf = df[predicate]
                results[ix] = func(subdf, **kwargs)

        return pd.Series(results)
