"""Dispatching code for Selection operations."""

from __future__ import annotations

import functools
import operator
from collections import defaultdict
from typing import TYPE_CHECKING, Any

import pandas as pd
from toolz import concatv, first

import ibis.expr.analysis as an
import ibis.expr.operations as ops
import ibis.expr.types as ir
from ibis.backends.base.df.scope import Scope
from ibis.backends.pandas.core import execute
from ibis.backends.pandas.dispatch import execute_node
from ibis.backends.pandas.execution import constants, util
from ibis.backends.pandas.execution.util import coerce_to_output

if TYPE_CHECKING:
    from collections.abc import Iterable

    from ibis.backends.base.df.timecontext import TimeContext


def compute_projection(
    node: ops.Node,
    parent: ops.Selection,
    data: pd.DataFrame,
    scope: Scope | None = None,
    timecontext: TimeContext | None = None,
    **kwargs: Any,
):
    """Compute a projection.

    `ibis.expr.types.Scalar` instances occur when a specific column projection
    is a window operation.
    """
    if isinstance(node, ops.TableNode):
        if node == parent.table:
            return data

        assert isinstance(parent.table, ops.Join)
        assert node in (parent.table.left, parent.table.right)

        mapping = remap_overlapping_column_names(
            parent.table,
            root_table=node,
            data_columns=frozenset(data.columns),
        )
        return map_new_column_names_to_data(mapping, data)
    elif isinstance(node, ops.Value):
        name = node.name
        assert name is not None, "Value selection name is None"

        if node.shape.is_scalar():
            data_columns = frozenset(data.columns)

            if scope is None:
                scope = Scope()

            scope = scope.merge_scopes(
                Scope(
                    {
                        t: map_new_column_names_to_data(
                            remap_overlapping_column_names(
                                parent.table, t, data_columns
                            ),
                            data,
                        )
                    },
                    timecontext,
                )
                for t in an.find_immediate_parent_tables(node)
            )
            scalar = execute(node, scope=scope, **kwargs)
            result = pd.Series([scalar], name=name).repeat(len(data.index))
            result.index = data.index
            return result
        else:
            if isinstance(node, ops.TableColumn):
                if name in data:
                    return data[name].rename(name)

                if not isinstance(parent.table, ops.Join):
                    raise KeyError(name)

                suffix = util.get_join_suffix_for_op(node, parent.table)
                return data.loc[:, name + suffix].rename(name)

            data_columns = frozenset(data.columns)

            scope = scope.merge_scopes(
                Scope(
                    {
                        t: map_new_column_names_to_data(
                            remap_overlapping_column_names(
                                parent.table, t, data_columns
                            ),
                            data,
                        )
                    },
                    timecontext,
                )
                for t in an.find_immediate_parent_tables(node)
            )

            result = execute(node, scope=scope, timecontext=timecontext, **kwargs)
            return coerce_to_output(result, node, data.index)
    else:
        raise TypeError(node)


def remap_overlapping_column_names(table, root_table, data_columns):
    """Return a mapping of suffixed column names to column names without suffixes.

    Parameters
    ----------
    table : TableNode
        The ``TableNode`` we're selecting from.
    root_table : TableNode
        The root table of the expression we're selecting from.
    data_columns
        The available columns to select from

    Returns
    -------
    dict[str, str]
        A mapping from possibly-suffixed column names to column names without
        suffixes.
    """
    if not isinstance(table, ops.Join):
        return None

    left_root, right_root = an.find_immediate_parent_tables([table.left, table.right])
    suffixes = {
        left_root: constants.LEFT_JOIN_SUFFIX,
        right_root: constants.RIGHT_JOIN_SUFFIX,
    }

    # if we're selecting from the root table and that's not the left or right
    # child, don't add a suffix
    #
    # this can happen when selecting directly from a join as opposed to
    # explicitly referencing the left or right tables
    #
    # we use setdefault here because the root_table can be the left/right table
    # which we may have already put into `suffixes`
    suffixes.setdefault(root_table, "")

    suffix = suffixes[root_table]

    column_names = [
        ({name, f"{name}{suffix}"} & data_columns, name)
        for name in root_table.schema.names
    ]
    mapping = {
        first(col_name): final_name for col_name, final_name in column_names if col_name
    }
    return mapping


def map_new_column_names_to_data(mapping, df):
    if mapping:
        return df.loc[:, mapping.keys()].rename(columns=mapping)
    return df


def _compute_predicates(
    table_op: ops.TableNode,
    predicates: Iterable[ir.BooleanColumn],
    data: pd.DataFrame,
    scope: Scope,
    timecontext: TimeContext | None,
    **kwargs: Any,
) -> pd.Series:
    """Compute the predicates for a table operation.

    This handles the cases where `predicates` are computed columns, in addition
    to the simple case of named columns coming directly from the input table.
    """
    for predicate in predicates:
        # Map each root table of the predicate to the data so that we compute
        # predicates on the result instead of any left or right tables if the
        # Selection is on a Join. Project data to only include columns from
        # the root table.
        root_tables = an.find_immediate_parent_tables(predicate)

        # handle suffixes
        data_columns = frozenset(data.columns)

        additional_scope = Scope()
        for root_table in root_tables:
            mapping = remap_overlapping_column_names(table_op, root_table, data_columns)
            new_data = map_new_column_names_to_data(mapping, data)
            additional_scope = additional_scope.merge_scope(
                Scope({root_table: new_data}, timecontext)
            )

        scope = scope.merge_scope(additional_scope)
        yield execute(predicate, scope=scope, **kwargs)


def build_df_from_selection(
    selections: list[ops.Value],
    data: pd.DataFrame,
    table: ops.Node,
) -> pd.DataFrame:
    """Build up a df by doing direct selections, renaming if necessary.

    Special logic for:
    - Joins where suffixes have been added to column names
    - Cases where new columns are created and selected.
    """
    cols = defaultdict(list)

    for node in selections:
        selection = node.name
        if selection not in data:
            if not isinstance(table, ops.Join):
                raise KeyError(selection)
            join_suffix = util.get_join_suffix_for_op(node, table)
            if selection + join_suffix not in data:
                raise KeyError(selection)
            selection += join_suffix
        cols[selection].append(node.name)

    result = data[list(cols.keys())]

    renamed_cols = {}
    for from_col, to_cols in cols.items():
        if len(to_cols) == 1 and from_col != to_cols[0]:
            renamed_cols[from_col] = to_cols[0]
        else:
            for new_col in to_cols:
                if from_col != new_col:
                    result[new_col] = result[from_col]

    if renamed_cols:
        result = result.rename(columns=renamed_cols)

    return result


def build_df_from_projection(
    selection_exprs: list[ir.Expr],
    op: ops.Selection,
    data: pd.DataFrame,
    **kwargs,
) -> pd.DataFrame:
    data_pieces = [
        compute_projection(node, op, data, **kwargs) for node in selection_exprs
    ]

    new_pieces = [
        piece.reset_index(level=list(range(1, piece.index.nlevels)), drop=True)
        if piece.index.nlevels > 1
        else piece
        for piece in data_pieces
    ]
    # Result series might be trimmed by time context, thus index may
    # have changed. To concat rows properly, we first `sort_index` on
    # each pieces then assign data index manually to series
    #
    # If cardinality changes (e.g. unnest/explode), trying to do this
    # won't work so don't try?
    for i, piece in enumerate(new_pieces):
        new_pieces[i] = piece.sort_index()
        if len(new_pieces[i].index) == len(data.index):
            new_pieces[i].index = data.index

    return pd.concat(new_pieces, axis=1)


@execute_node.register(ops.Selection, pd.DataFrame)
def execute_selection_dataframe(
    op,
    data,
    scope: Scope,
    timecontext: TimeContext | None,
    **kwargs,
):
    result = data

    # Build up the individual pandas structures from column expressions
    if op.selections:
        if all(isinstance(s, ops.TableColumn) for s in op.selections):
            result = build_df_from_selection(op.selections, data, op.table)
        else:
            result = build_df_from_projection(
                op.selections,
                op,
                data,
                scope=scope,
                timecontext=timecontext,
                **kwargs,
            )

    if op.predicates:
        predicates = _compute_predicates(
            op.table, op.predicates, data, scope, timecontext, **kwargs
        )
        predicate = functools.reduce(operator.and_, predicates)
        assert len(predicate) == len(
            result
        ), "Selection predicate length does not match underlying table"
        result = result.loc[predicate]

    if op.sort_keys:
        result, grouping_keys, ordering_keys = util.compute_sorted_frame(
            result,
            order_by=op.sort_keys,
            scope=scope,
            timecontext=timecontext,
            **kwargs,
        )
    else:
        grouping_keys = ordering_keys = ()

    # return early if we do not have any temporary grouping or ordering columns
    assert not grouping_keys, "group by should never show up in Selection"
    if not ordering_keys:
        return result

    # create a sequence of columns that we need to drop
    temporary_columns = pd.Index(concatv(grouping_keys, ordering_keys)).difference(
        data.columns
    )

    # no reason to call drop if we don't need to
    if temporary_columns.empty:
        return result

    # drop every temporary column we created for ordering or grouping
    return result.drop(temporary_columns, axis=1)
