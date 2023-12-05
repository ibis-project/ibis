from __future__ import annotations

import operator
from functools import reduce

import pandas as pd

import ibis.expr.operations as ops
from ibis.backends.pandas.executor.core import execute
from ibis.backends.pandas.executor.utils import asframe, asseries
from ibis.backends.pandas.rewrites import (
    PandasAggregate,
    PandasJoin,
    PandasReduce,
    PandasRename,
)
from ibis.util import gen_name

################## PANDAS SPECIFIC NODES ######################


@execute.register(ops.DatabaseTable)
def execute_database_table(op, name, schema, source, namespace):
    return source.dictionary[name]


@execute.register(ops.InMemoryTable)
def execute_in_memory_table(op, name, schema, data):
    return data.to_frame()


@execute.register(ops.DummyTable)
def execute_dummy_table(op, values):
    df, _ = asframe(values)
    return df


@execute.register(ops.Limit)
def execute_limit(op, parent, n, offset):
    if n is None:
        return parent.iloc[offset:]
    else:
        return parent.iloc[offset : offset + n]


@execute.register(ops.Sample)
def execute_sample(op, parent, fraction, method, seed):
    return parent.sample(frac=fraction, random_state=seed)


@execute.register(ops.Filter)
def execute_filter(op, parent, predicates):
    if predicates:
        pred = reduce(operator.and_, predicates)
        if len(pred) != len(parent):
            raise RuntimeError(
                "Selection predicate length does not match underlying table"
            )
        parent = parent.loc[pred].reset_index(drop=True)
    return parent


@execute.register(PandasRename)
def execute_rename(op, parent, mapping):
    return parent.rename(columns=mapping)


@execute.register(PandasJoin)
def execute_join(op, left, right, left_on, right_on, how):
    # broadcast predicates if they are scalar values
    left_size = len(left)
    left_on = [asseries(v, left_size) for v in left_on]
    right_size = len(right)
    right_on = [asseries(v, right_size) for v in right_on]

    if how == "cross":
        assert not left_on and not right_on
        return pd.merge(left, right, how="cross")
    elif how == "anti":
        df = pd.merge(
            left, right, how="outer", left_on=left_on, right_on=right_on, indicator=True
        )
        df = df[df["_merge"] == "left_only"]
        return df.drop(columns=["_merge"])
    elif how == "semi":
        mask = asseries(True, left_size)
        for left_pred, right_pred in zip(left_on, right_on):
            mask = mask & left_pred.isin(right_pred)
        return left[mask]
    elif how == "asof":
        df = pd.merge_asof(left, right, left_on=left_on, right_on=right_on)
        return df
    else:
        df = left.merge(right, how=how, left_on=left_on, right_on=right_on)
        return df.drop(columns=[f"key_{i}" for i in range(len(left_on))])


@execute.register(ops.Union)
def execute_union(op, left, right, distinct):
    result = pd.concat([left, right], axis=0)
    return result.drop_duplicates() if distinct else result


@execute.register(ops.Intersection)
def execute_intersection_dataframe_dataframe(op, left, right, distinct):
    if not distinct:
        raise NotImplementedError(
            "`distinct=False` is not supported by the pandas backend"
        )
    return left.merge(right, on=list(left.columns), how="inner")


@execute.register(ops.Distinct)
def execute_distinct(op, parent):
    return parent.drop_duplicates()


@execute.register(ops.Project)
def execute_project(op, parent, values):
    df, _ = asframe(values)
    return df


@execute.register(ops.Sort)
def execute_sort(op, parent, keys):
    # 1. add sort key columns to the dataframe if they are not already present
    # 2. sort the dataframe using those columns
    # 3. drop the sort key columns

    names = []
    newcols = {}
    for key, keycol in zip(op.keys, keys):
        if not isinstance(key, ops.Field):
            name = gen_name("sort_key")
            newcols[name] = keycol
        names.append(name)

    result = parent.assign(**newcols)
    ascending = [key.ascending for key in op.keys]
    result = result.sort_values(by=names, ascending=ascending, ignore_index=True)

    return result.drop(newcols.keys(), axis=1)


@execute.register(ops.DropNa)
def execute_dropna(op, parent, how, subset):
    if op.subset is not None:
        subset = [col.name for col in op.subset]
    else:
        subset = None
    return parent.dropna(how=how, subset=subset)


@execute.register(PandasReduce)
def execute_pandas_reduce(op, parent, metrics):
    results = {k: v(parent) for k, v in metrics.items()}
    combined, _ = asframe(results)
    return combined


@execute.register(PandasAggregate)
def execute_pandas_aggregate(op, parent, groups, metrics):
    parent = parent.groupby([name for name, col in groups.items()])
    metrics = {k: parent.apply(v) for k, v in metrics.items()}

    result = pd.concat(metrics, axis=1).reset_index()
    return result


@execute.register(ops.InValues)
def execute_in_values(op, value, options):
    if isinstance(value, pd.Series):
        return value.isin(options)
    else:
        return value in options


@execute.register(ops.InSubquery)
def execute_in_subquery(op, rel, needle):
    first_column = rel.iloc[:, 0]
    if isinstance(needle, pd.Series):
        return needle.isin(first_column)
    else:
        return needle in first_column
