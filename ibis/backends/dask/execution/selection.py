"""Dispatching code for Selection operations.
"""

from __future__ import absolute_import

import functools
import operator
from typing import Optional

import dask.dataframe as dd
import numpy as np
import pandas
from toolz import concatv

import ibis.expr.operations as ops
import ibis.expr.types as ir
from ibis.backends.pandas.execution.selection import (
    _compute_predicates,
    compute_projection,
    compute_projection_table_expr,
    execute,
    execute_node,
    map_new_column_names_to_data,
    remap_overlapping_column_names,
)
from ibis.expr.scope import Scope
from ibis.expr.typing import TimeContext

from ..execution import constants


@compute_projection.register(ir.ScalarExpr, ops.Selection, dd.DataFrame)
def compute_projection_scalar_expr(
    expr,
    parent,
    data,
    scope: Scope = None,
    timecontext: Optional[TimeContext] = None,
    **kwargs,
):
    name = expr._name
    assert name is not None, 'Scalar selection name is None'

    op = expr.op()
    parent_table_op = parent.table.op()

    data_columns = frozenset(data.columns)

    scope = scope.merge_scopes(
        Scope(
            {
                t: map_new_column_names_to_data(
                    remap_overlapping_column_names(
                        parent_table_op, t, data_columns
                    ),
                    data,
                )
            },
            timecontext,
        )
        for t in op.root_tables()
    )
    scalar = execute(expr, scope=scope, **kwargs)
    result = pandas.Series([scalar], name=name).repeat(len(data.index))
    result.index = data.index
    return dd.from_pandas(result, npartitions=data.npartitions)


@compute_projection.register(ir.ColumnExpr, ops.Selection, dd.DataFrame)
def compute_projection_column_expr(
    expr,
    parent,
    data,
    scope: Scope,
    timecontext: Optional[TimeContext],
    **kwargs,
):
    result_name = getattr(expr, '_name', None)
    op = expr.op()
    parent_table_op = parent.table.op()

    if isinstance(op, ops.TableColumn):
        # slightly faster path for simple column selection
        name = op.name

        if name in data:
            return data[name].rename(result_name or name)

        if not isinstance(parent_table_op, ops.Join):
            raise KeyError(name)
        (root_table,) = op.root_tables()
        left_root, right_root = ops.distinct_roots(
            parent_table_op.left, parent_table_op.right
        )
        suffixes = {
            left_root: constants.LEFT_JOIN_SUFFIX,
            right_root: constants.RIGHT_JOIN_SUFFIX,
        }
        return data.loc[:, name + suffixes[root_table]].rename(
            result_name or name
        )

    data_columns = frozenset(data.columns)

    scope = scope.merge_scopes(
        Scope(
            {
                t: map_new_column_names_to_data(
                    remap_overlapping_column_names(
                        parent_table_op, t, data_columns
                    ),
                    data,
                )
            },
            timecontext,
        )
        for t in op.root_tables()
    )

    result = execute(expr, scope=scope, timecontext=timecontext, **kwargs)
    assert result_name is not None, 'Column selection name is None'
    if np.isscalar(result):
        series = dd.from_array(np.repeat(result, len(data.index)))
        series.name = result_name
        series.index = data.index
        return series
    return result.rename(result_name)


compute_projection.register(ir.TableExpr, ops.Selection, dd.DataFrame)(
    compute_projection_table_expr
)


# TODO - sorting - #2553
@execute_node.register(ops.Selection, dd.DataFrame)
def execute_selection_dataframe(
    op, data, scope: Scope, timecontext: Optional[TimeContext], **kwargs
):
    selections = op.selections
    predicates = op.predicates
    sort_keys = op.sort_keys
    result = data

    # Build up the individual dask structures from column expressions
    if selections:
        data_pieces = []
        for selection in selections:
            dask_object = compute_projection(
                selection,
                op,
                data,
                scope=scope,
                timecontext=timecontext,
                **kwargs,
            )
            data_pieces.append(dask_object)

        new_pieces = [piece for piece in data_pieces]
        result = dd.concat(new_pieces, axis=1)

    if predicates:
        predicates = _compute_predicates(
            op.table.op(), predicates, data, scope, timecontext, **kwargs
        )
        predicate = functools.reduce(operator.and_, predicates)
        assert len(predicate) == len(
            result
        ), 'Selection predicate length does not match underlying table'
        result = result.loc[predicate]

    if sort_keys:
        raise NotImplementedError(
            "Sorting is not implemented for the Dask backend"
        )
        # result, grouping_keys, ordering_keys = util.compute_sorted_frame(
        #     result,
        #     order_by=sort_keys,
        #     scope=scope,
        #     timecontext=timecontext,
        #     **kwargs,
        # )
    else:
        grouping_keys = ordering_keys = ()

    # return early if we do not have any temporary grouping or ordering columns
    assert not grouping_keys, 'group by should never show up in Selection'
    if not ordering_keys:
        return result

    # create a sequence of columns that we need to drop
    temporary_columns = pandas.Index(
        concatv(grouping_keys, ordering_keys)
    ).difference(data.columns)

    # no reason to call drop if we don't need to
    if temporary_columns.empty:
        return result

    # drop every temporary column we created for ordering or grouping
    return result.drop(temporary_columns, axis=1)
