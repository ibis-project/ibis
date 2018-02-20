"""Dispatching code for Selection operations.
"""

from __future__ import absolute_import

import itertools
import operator

from collections import OrderedDict

import pandas as pd

import toolz

from multipledispatch import Dispatcher

import ibis.expr.types as ir
import ibis.expr.operations as ops

from ibis.compat import functools

from ibis.pandas.dispatch import execute, execute_node
from ibis.pandas.execution import constants, util


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
scope : dict, optional

Returns
-------
value : scalar, pd.Series, pd.DataFrame

Notes
-----
:class:`~ibis.expr.types.ScalarExpr` instances occur when a specific column
projection is a window operation.
"""
)


@compute_projection.register(ir.ScalarExpr, ops.Selection, pd.DataFrame)
def compute_projection_scalar_expr(expr, parent, data, scope=None, **kwargs):
    name = expr._name
    assert name is not None, 'Scalar selection name is None'

    op = expr.op()
    parent_table_op = parent.table.op()

    data_columns = frozenset(data.columns)

    additional_scope = OrderedDict(
        (
            t,
            map_new_column_names_to_data(
                remap_overlapping_column_names(
                    parent_table_op, t, data_columns
                ),
                data
            )
        ) for t in op.root_tables()
    )

    new_scope = toolz.merge(scope, additional_scope, factory=OrderedDict)
    result = execute(expr, new_scope, **kwargs)
    return pd.Series([result], name=name, index=data.index)


@compute_projection.register(ir.ColumnExpr, ops.Selection, pd.DataFrame)
def compute_projection_column_expr(expr, parent, data, scope=None, **kwargs):
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

        root_table, = op.root_tables()
        left_root, right_root = ir.distinct_roots(
            parent_table_op.left, parent_table_op.right
        )
        suffixes = {left_root: constants.LEFT_JOIN_SUFFIX,
                    right_root: constants.RIGHT_JOIN_SUFFIX}
        return data.loc[:, name + suffixes[root_table]].rename(
            result_name or name
        )

    data_columns = frozenset(data.columns)
    additional_scope = {
        t: map_new_column_names_to_data(
            remap_overlapping_column_names(parent_table_op, t, data_columns),
            data
        ) for t in op.root_tables()
    }

    new_scope = toolz.merge(scope, additional_scope)
    result = execute(expr, new_scope, **kwargs)
    assert result_name is not None, 'Column selection name is None'
    return result.rename(result_name)


@compute_projection.register(ir.TableExpr, ops.Selection, pd.DataFrame)
def compute_projection_table_expr(expr, parent, data, **kwargs):
    if expr is parent.table:
        return data

    parent_table_op = parent.table.op()
    assert isinstance(parent_table_op, ops.Join)
    assert (expr.equals(parent_table_op.left) or
            expr.equals(parent_table_op.right))

    mapping = remap_overlapping_column_names(
        parent_table_op,
        root_table=expr.op(),
        data_columns=frozenset(data.columns)
    )
    return map_new_column_names_to_data(mapping, data)


@compute_projection.register(object, ops.Selection, pd.DataFrame)
def compute_projection_default(op, parent, data, **kwargs):
    raise TypeError(
        "Don't know how to compute projection of {}".format(type(op).__name__)
    )


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

    left_root, right_root = ir.distinct_roots(table_op.left, table_op.right)
    suffixes = {left_root: constants.LEFT_JOIN_SUFFIX,
                right_root: constants.RIGHT_JOIN_SUFFIX}
    column_names = [
        ({name, name + suffixes[root_table]} & data_columns, name)
        for name in root_table.schema.names
    ]
    mapping = OrderedDict(
        (toolz.first(col_name), final_name)
        for col_name, final_name in column_names if col_name
    )
    return mapping


def map_new_column_names_to_data(mapping, df):
    if mapping is not None:
        return df.loc[:, mapping.keys()].rename(columns=mapping)
    return df


def _compute_predicates(table_op, predicates, data, scope, **kwargs):
    """Compute the predicates for a table operation.

    Parameters
    ----------
    table_op : TableNode
    predicates : List[ir.ColumnExpr]
    data : pd.DataFrame
    scope : dict
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
        additional_scope = {}
        data_columns = frozenset(data.columns)

        for root_table in root_tables:
            mapping = remap_overlapping_column_names(
                table_op, root_table, data_columns
            )
            if mapping is not None:
                new_data = data.loc[:, mapping.keys()].rename(columns=mapping)
            else:
                new_data = data
            additional_scope[root_table] = new_data

        new_scope = toolz.merge(scope, additional_scope)
        yield execute(predicate, new_scope, **kwargs)


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
"""
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
    return list(toolz.unique(
        itertools.chain(
            toolz.unique(physical_tables(join.left.op()), key=id),
            toolz.unique(physical_tables(join.right.op()), key=id)
        ),
        key=id,
    ))


@physical_tables.register(ops.Node)
def physical_tables_node(node):
    # Iterative case. Any other Node's physical roots are the unique physical
    # roots of that Node's root tables.
    tables = toolz.concat(map(physical_tables, node.root_tables()))
    return list(toolz.unique(tables, key=id))


@execute_node.register(ops.Selection, pd.DataFrame)
def execute_selection_dataframe(op, data, scope=None, **kwargs):
    selections = op.selections
    predicates = op.predicates
    sort_keys = op.sort_keys
    result = data

    # Build up the individual pandas structures from column expressions
    if selections:
        data_pieces = []

        for selection in selections:
            pandas_object = compute_projection(
                selection, op, data, scope=scope, **kwargs
            )
            data_pieces.append(pandas_object)
        result = pd.concat(data_pieces, axis=1)

    if predicates:
        predicates = _compute_predicates(
            op.table.op(), predicates, data, scope, **kwargs
        )
        predicate = functools.reduce(operator.and_, predicates)
        assert len(predicate) == len(result), \
            'Selection predicate length does not match underlying table'
        result = result.loc[predicate]

    if sort_keys:
        result = util.compute_sorted_frame(sort_keys, result, **kwargs)
    return result.reset_index(drop=True)
