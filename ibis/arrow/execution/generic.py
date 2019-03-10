"""
Generic execution functions for the arrow backend.
"""

import operator
import six
import toolz

import pandas as pd
from pandas.core.groupby import DataFrameGroupBy, SeriesGroupBy

import numpy as np
import pyarrow as pa

import ibis.expr.operations as ops
import ibis.expr.datatypes as dt
from ibis.compat import functools, map, zip

from arrow.client import ArrowTable, ArrowClient
from arrow.core import execute, integer_types, numeric_types
from arrow.dispatch import execute_node, execute_literal
from arrow.execution import constants
from arrow.execution.selection import get_index_by_name


@execute_node.register(ArrowTable, ArrowClient)
def execute_database_table_client(op, client, **kwargs):
    """Accesses the table of the corresponding client."""
    return client.dictionary[op.name]


@execute_literal.register(ops.Literal, object, dt.DataType)
def execute_node_literal_value_datatype(op, value, datatype, **kwargs):
    """Returns the literal value by default."""
    return value


@execute_node.register(ops.Aggregation, pa.RecordBatch)
def execute_aggregation_record_batch(op, data, scope=None, **kwargs):
    """Performs an aggregation on a record batch."""
    assert op.metrics, 'no metrics found during aggregation execution'

    if op.sort_keys:
        raise NotImplementedError(
            'sorting on aggregations not yet implemented'
        )

    predicates = op.predicates
    if predicates:
        predicate = functools.reduce(
            operator.and_,
            (execute(p, scope=scope, **kwargs) for p in predicates)
        )
        data = data.loc[predicate]

    columns = {}

    if op.by:
        grouping_key_pairs = list(
            zip(op.by, map(operator.methodcaller('op'), op.by))
        )
        grouping_keys = [
            by_op.name if isinstance(by_op, ops.TableColumn)
            else execute(by, scope=scope, **kwargs).rename(by.get_name())
            for by, by_op in grouping_key_pairs
        ]
        columns.update(
            (by_op.name, by.get_name()) for by, by_op in grouping_key_pairs
            if hasattr(by_op, 'name')
        )
        source = data.to_pandas().groupby(grouping_keys)
    else:
        source = data.to_pandas()

    new_scope = toolz.merge(scope, {op.table.op(): source})
    pieces = [
        pd.Series(
            execute(metric, scope=new_scope, **kwargs), name=metric.get_name())
        for metric in op.metrics
    ]

    # group by always needs a reset to get the grouping key back as a column
    result = pd.concat(pieces, axis=1).reset_index()
    result.columns = [columns.get(c, c) for c in result.columns]

    result = pa.RecordBatch.from_pandas(result)

    if op.having:
        # .having(...) is only accessible on groupby, so this should never
        # raise
        if not op.by:
            raise ValueError(
                'Filtering out aggregation values is not allowed without at '
                'least one grouping key'
            )

        # TODO(phillipc): Don't recompute identical subexpressions
        predicate = functools.reduce(
            operator.and_,
            (execute(having, scope=new_scope, **kwargs)
             for having in op.having)
        )
        assert len(predicate) == len(result), \
            'length of predicate does not match length of RecordBatch'
        result = result.loc[predicate.values]
    return result


@execute_node.register(ops.BinaryOp, pa.Array, pa.Array)
@execute_node.register(ops.BinaryOp, np.ndarray, np.ndarray)
@execute_node.register(
    (ops.NumericBinaryOp, ops.LogicalBinaryOp, ops.Comparison),
    numeric_types,
    pa.Array,
)
@execute_node.register(
    (ops.NumericBinaryOp, ops.LogicalBinaryOp, ops.Comparison),
    pa.Array,
    numeric_types,
)
@execute_node.register(
    (ops.NumericBinaryOp, ops.LogicalBinaryOp, ops.Comparison),
    numeric_types,
    numeric_types,
)
@execute_node.register(
    (ops.Comparison, ops.Add, ops.Multiply), pa.Array, six.string_types)
@execute_node.register(
    (ops.Comparison, ops.Add, ops.Multiply), six.string_types, pa.Array)
@execute_node.register(
    (ops.Comparison, ops.Add), six.string_types, six.string_types)
@execute_node.register(ops.Multiply, integer_types, six.string_types)
@execute_node.register(ops.Multiply, six.string_types, integer_types)
def execute_binary_op(op, left, right, **kwargs):
    """Performs a binary operation a columns."""
    op_type = type(op)
    try:
        operation = constants.BINARY_OPERATIONS[op_type]
    except KeyError:
        raise NotImplementedError(
            'Binary operation {} not implemented'.format(op_type.__name__)
        )
    else:
        if issubclass(type(left), pa.lib.Array):
            left = left.to_pandas()
        if issubclass(type(right), pa.lib.Array):
            right = right.to_pandas()

        return operation(left, right)


@execute_node.register(ops.Reduction, SeriesGroupBy, type(None))
def execute_reduction_series_groupby(
        op, data, mask, aggcontext=None, **kwargs
):
    """Performs a grouping action on a column."""
    return aggcontext.agg(data, type(op).__name__.lower())


@execute_node.register(ops.TableColumn, (pd.DataFrame, DataFrameGroupBy))
def execute_table_column_df_or_df_groupby(op, data, **kwargs):
    """Accesses the column of a dataframe."""
    return data[op.name]


@execute_node.register(ops.TableColumn, pa.RecordBatch)
def execute_table_column_df_or_df_groupby(op, data, **kwargs):
    """Accesses the column of a record batch."""
    return data[get_index_by_name(data, op.name)]
