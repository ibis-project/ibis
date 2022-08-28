"""Dispatching code for Selection operations.
"""


import functools
import operator
from typing import List, Optional

import dask.dataframe as dd
import pandas
from toolz import concatv

import ibis.expr.operations as ops
import ibis.expr.types as ir
from ibis.backends.dask.core import execute
from ibis.backends.dask.dispatch import execute_node
from ibis.backends.dask.execution.util import (
    add_partitioned_sorted_column,
    coerce_to_output,
    compute_sorted_frame,
    is_row_order_preserving,
)
from ibis.backends.pandas.execution.selection import (
    build_df_from_selection,
    compute_projection,
    compute_projection_table_expr,
    map_new_column_names_to_data,
    remap_overlapping_column_names,
)
from ibis.backends.pandas.execution.util import get_join_suffix_for_op
from ibis.expr.scope import Scope
from ibis.expr.typing import TimeContext


@compute_projection.register(ir.Scalar, ops.Selection, dd.DataFrame)
def compute_projection_scalar_expr(
    expr,
    parent,
    data,
    scope: Scope,
    timecontext: Optional[TimeContext] = None,
    **kwargs,
):
    name = expr.get_name()
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
    return data.assign(**{name: scalar})[name]


@compute_projection.register(ir.Column, ops.Selection, dd.DataFrame)
def compute_projection_column_expr(
    expr,
    parent,
    data,
    scope: Scope,
    timecontext: Optional[TimeContext],
    **kwargs,
):
    result_name = expr._safe_name
    op = expr.op()
    parent_table_op = parent.table.op()

    if isinstance(op, ops.TableColumn):
        # slightly faster path for simple column selection
        name = op.name

        if name in data:
            return data[name].rename(result_name or name)

        if not isinstance(parent_table_op, ops.Join):
            raise KeyError(name)
        suffix = get_join_suffix_for_op(op, parent_table_op)
        return data.loc[:, name + suffix].rename(result_name or name)

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
    result = coerce_to_output(result, expr, data.index)
    return result


compute_projection.register(ir.Table, ops.Selection, dd.DataFrame)(
    compute_projection_table_expr
)


def build_df_from_projection(
    selection_exprs: List[ir.Expr],
    op: ops.Selection,
    data: dd.DataFrame,
    **kwargs,
) -> dd.DataFrame:
    """
    Build up a df from individual pieces by dispatching to `compute_projection`
    for each expression.
    """
    # Fast path for when we're assigning columns into the same table.
    if (selection_exprs[0] is op.table) and all(
        is_row_order_preserving(selection_exprs[1:])
    ):
        for expr in selection_exprs[1:]:
            projection = compute_projection(expr, op, data, **kwargs)
            if isinstance(projection, dd.Series):
                data = data.assign(**{projection.name: projection})
            else:
                data = data.assign(
                    **{c: projection[c] for c in projection.columns}
                )
        return data

    # Slow path when we cannot do direct assigns
    # Create a unique row identifier and set it as the index. This is
    # used in dd.concat to merge the pieces back together.
    data = add_partitioned_sorted_column(data)
    data_pieces = [
        compute_projection(expr, op, data, **kwargs)
        for expr in selection_exprs
    ]

    return dd.concat(data_pieces, axis=1).reset_index(drop=True)


@execute_node.register(ops.Selection, dd.DataFrame, tuple, tuple, tuple)
def execute_selection_dataframe(
    op,
    data,
    selections,
    predicates,
    sort_keys,
    scope: Scope,
    timecontext: Optional[TimeContext],
    **kwargs,
):
    result = data

    if predicates:
        predicates = _compute_predicates(
            op.table.op(), predicates, data, scope, timecontext, **kwargs
        )
        predicate = functools.reduce(operator.and_, predicates)
        result = result.loc[predicate]

    if selections:
        # if we are just performing select operations we can do a direct
        # selection
        if all(isinstance(s.op(), ops.TableColumn) for s in selections):
            result = build_df_from_selection(selections, result, op.table.op())
        else:
            result = build_df_from_projection(
                selections,
                op,
                result,
                scope=scope,
                timecontext=timecontext,
                **kwargs,
            )

    if sort_keys:
        if len(sort_keys) > 1:
            raise NotImplementedError(
                """
                Multi-key sorting is not implemented for the Dask backend
                """
            )
        sort_key = sort_keys[0]
        ascending = getattr(sort_key.op(), 'ascending', True)
        if not ascending:
            raise NotImplementedError(
                "Descending sort is not supported for the Dask backend"
            )
        result = compute_sorted_frame(
            result,
            order_by=sort_key,
            scope=scope,
            timecontext=timecontext,
            **kwargs,
        )

        return result
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


def _compute_predicates(
    table_op,
    predicates,
    data,
    scope: Scope,
    timecontext: Optional[TimeContext],
    **kwargs,
):
    """Compute the predicates for a table operation.

    Parameters
    ----------
    table_op : TableNode
    predicates : List[ir.Column]
    data : pd.DataFrame
    scope : Scope
    timecontext: Optional[TimeContext]
    kwargs : dict

    Returns
    -------
    computed_predicate : pd.Series[bool]

    Notes
    -----
    This handles the cases where the predicates are computed columns, in
    addition to the simple case of named columns coming directly from the input
    table.
    """
    for predicate in predicates:
        # Map each root table of the predicate to the data so that we compute
        # predicates on the result instead of any left or right tables if the
        # Selection is on a Join. Project data to only inlude columns from
        # the root table.
        root_tables = predicate.op().root_tables()

        # handle suffixes
        data_columns = frozenset(data.columns)

        additional_scope = Scope()
        for root_table in root_tables:
            mapping = remap_overlapping_column_names(
                table_op, root_table, data_columns
            )
            if mapping is not None:
                new_data = data.loc[:, mapping.keys()].rename(columns=mapping)
            else:
                new_data = data
            additional_scope = additional_scope.merge_scope(
                Scope({root_table: new_data}, timecontext)
            )

        scope = scope.merge_scope(additional_scope)
        yield execute(predicate, scope=scope, **kwargs)
