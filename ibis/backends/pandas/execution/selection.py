"""Dispatching code for Selection operations.
"""

from __future__ import absolute_import

import functools
import operator
from collections import OrderedDict
from operator import methodcaller
from typing import Optional

import pandas as pd
from multipledispatch import Dispatcher
from toolz import compose, concat, concatv, first, unique

import ibis.expr.operations as ops
import ibis.expr.types as ir
from ibis.expr.scope import Scope
from ibis.expr.typing import TimeContext

from ..core import execute
from ..dispatch import execute_node
from ..execution import constants, util
from ..execution.util import coerce_to_output

compute_projection = Dispatcher(
    'compute_projection',
    doc="""\
Compute a projection, dispatching on whether we're computing a scalar, column,
or table expression.

Parameters
----------
expr : Union[ir.ScalarExpr, ir.ColumnExpr, ir.TableExpr]
parent : ops.Selection
data : pd.DataFrame
scope : Scope
timecontext:Optional[TimeContext]

Returns
-------
value : scalar, pd.Series, pd.DataFrame

Notes
-----
:class:`~ibis.expr.types.ScalarExpr` instances occur when a specific column
projection is a window operation.
""",
)


@compute_projection.register(ir.ScalarExpr, ops.Selection, pd.DataFrame)
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
    result = pd.Series([scalar], name=name).repeat(len(data.index))
    result.index = data.index
    return result


@compute_projection.register(ir.ColumnExpr, ops.Selection, pd.DataFrame)
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

    result = coerce_to_output(
        execute(expr, scope=scope, timecontext=timecontext, **kwargs),
        expr,
        data.index,
    )
    assert result_name is not None, 'Column selection name is None'
    return result


@compute_projection.register(ir.TableExpr, ops.Selection, pd.DataFrame)
def compute_projection_table_expr(expr, parent, data, **kwargs):
    if expr is parent.table:
        return data

    parent_table_op = parent.table.op()
    assert isinstance(parent_table_op, ops.Join)
    assert expr.equals(parent_table_op.left) or expr.equals(
        parent_table_op.right
    )

    mapping = remap_overlapping_column_names(
        parent_table_op,
        root_table=expr.op(),
        data_columns=frozenset(data.columns),
    )
    return map_new_column_names_to_data(mapping, data)


def remap_overlapping_column_names(table_op, root_table, data_columns):
    """Return an ``OrderedDict`` mapping possibly suffixed column names to
    column names without suffixes.

    Parameters
    ----------
    table_op : TableNode
        The ``TableNode`` we're selecting from.
    root_table : TableNode
        The root table of the expression we're selecting from.
    data_columns : set or frozenset
        The available columns to select from

    Returns
    -------
    mapping : OrderedDict[str, str]
        A map from possibly-suffixed column names to column names without
        suffixes.
    """
    if not isinstance(table_op, ops.Join):
        return None

    left_root, right_root = ops.distinct_roots(table_op.left, table_op.right)
    suffixes = {
        left_root: constants.LEFT_JOIN_SUFFIX,
        right_root: constants.RIGHT_JOIN_SUFFIX,
    }
    column_names = [
        ({name, name + suffixes[root_table]} & data_columns, name)
        for name in root_table.schema.names
    ]
    mapping = OrderedDict(
        (first(col_name), final_name)
        for col_name, final_name in column_names
        if col_name
    )
    return mapping


def map_new_column_names_to_data(mapping, df):
    if mapping is not None:
        return df.loc[:, mapping.keys()].rename(columns=mapping)
    return df


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
    predicates : List[ir.ColumnExpr]
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


physical_tables = Dispatcher(
    'physical_tables',
    doc="""\
Return the underlying physical tables nodes of a
:class:`~ibis.expr.types.Node`.

Parameters
----------
op : ops.Node

Returns
-------
tables : List[ops.Node]
""",
)


@physical_tables.register(ops.Selection)
def physical_tables_selection(sel):
    return physical_tables(sel.table.op())


@physical_tables.register(ops.PhysicalTable)
def physical_tables_physical_table(t):
    # Base case. PhysicalTable nodes are their own root physical tables.
    return [t]


@physical_tables.register(ops.Join)
def physical_tables_join(join):
    # Physical roots of Join nodes are the unique physical roots of their
    # left and right TableNodes.
    func = compose(physical_tables, methodcaller('op'))
    return list(unique(concat(map(func, (join.left, join.right)))))


@physical_tables.register(ops.Node)
def physical_tables_node(node):
    # Iterative case. Any other Node's physical roots are the unique physical
    # roots of that Node's root tables.
    return list(unique(concat(map(physical_tables, node.root_tables()))))


@execute_node.register(ops.Selection, pd.DataFrame)
def execute_selection_dataframe(
    op, data, scope: Scope, timecontext: Optional[TimeContext], **kwargs
):
    selections = op.selections
    predicates = op.predicates
    sort_keys = op.sort_keys
    result = data

    # Build up the individual pandas structures from column expressions
    if selections:
        data_pieces = []
        for selection in selections:
            pandas_object = compute_projection(
                selection,
                op,
                data,
                scope=scope,
                timecontext=timecontext,
                **kwargs,
            )
            data_pieces.append(pandas_object)

        new_pieces = [
            piece.reset_index(
                level=list(range(1, piece.index.nlevels)), drop=True
            )
            if piece.index.nlevels > 1
            else piece
            for piece in data_pieces
        ]
        result = pd.concat(new_pieces, axis=1)

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
        result, grouping_keys, ordering_keys = util.compute_sorted_frame(
            result,
            order_by=sort_keys,
            scope=scope,
            timecontext=timecontext,
            **kwargs,
        )
    else:
        grouping_keys = ordering_keys = ()

    # return early if we do not have any temporary grouping or ordering columns
    assert not grouping_keys, 'group by should never show up in Selection'
    if not ordering_keys:
        return result

    # create a sequence of columns that we need to drop
    temporary_columns = pd.Index(
        concatv(grouping_keys, ordering_keys)
    ).difference(data.columns)

    # no reason to call drop if we don't need to
    if temporary_columns.empty:
        return result

    # drop every temporary column we created for ordering or grouping
    return result.drop(temporary_columns, axis=1)
