"""ClickHouse ibis expression to sqlglot compiler.

The compiler is built with a few `singledispatch` functions:

    1. `translate_rel` for compiling `ops.TableNode`s
    1. `translate_val` for compiling `ops.Value`s

## `translate`

### Node Implementation

There's a single `ops.Node` implementation for `ops.TableNode`s instances.

This function compiles each node in topological order. The topological sorting,
result caching, and iteration are all handled by
`ibis.expr.operations.core.Node.map`.
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Any

import sqlglot as sg

import ibis.expr.operations as ops
import ibis.expr.types as ir
from ibis.backends.clickhouse.compiler.relations import translate_rel
from ibis.backends.clickhouse.compiler.values import translate_val
from ibis.common.deferred import _
from ibis.expr.analysis import c, find_first_base_table, p, x, y
from ibis.expr.rewrites import rewrite_dropna, rewrite_fillna, rewrite_sample

if TYPE_CHECKING:
    from collections.abc import Mapping


def _translate_node(node, **kwargs):
    if isinstance(node, ops.Value):
        return translate_val(node, **kwargs)
    assert isinstance(node, ops.TableNode)
    return translate_rel(node, **kwargs)


def translate(op: ops.TableNode, params: Mapping[ir.Value, Any]) -> sg.exp.Expression:
    """Translate an ibis operation to a sqlglot expression.

    Parameters
    ----------
    op
        An ibis `TableNode`
    params
        A mapping of expressions to concrete values

    Returns
    -------
    sqlglot.expressions.Expression
        A sqlglot expression
    """

    gen_alias_index = itertools.count()

    def fn(node, _, **kwargs):
        result = _translate_node(node, **kwargs)

        # don't alias root nodes or value ops
        if node is op or isinstance(node, ops.Value):
            return result

        assert isinstance(node, ops.TableNode)

        alias_index = next(gen_alias_index)
        alias = f"t{alias_index:d}"

        try:
            return result.subquery(alias)
        except AttributeError:
            return sg.alias(result, alias)

    # substitute parameters immediately to avoid having to define a
    # ScalarParameter translation rule
    #
    # this lets us avoid threading `params` through every `translate_val` call
    # only to be used in the one place it would be needed: the ScalarParameter
    # `translate_val` rule
    params = {param.op(): value for param, value in params.items()}
    replace_literals = p.ScalarParameter(dtype=x) >> (
        lambda _, x: ops.Literal(value=params[_], dtype=x)
    )

    # replace the right side of InColumn into a scalar subquery for sql
    # backends
    replace_in_column_with_table_array_view = p.InColumn(..., y) >> _.copy(
        options=c.TableArrayView(
            c.Selection(table=lambda _, y: find_first_base_table(y), selections=(y,))
        ),
    )

    # replace any checks against an empty right side of the IN operation with
    # `False`
    replace_empty_in_values_with_false = p.InValues(..., ()) >> c.Literal(
        False, dtype="bool"
    )

    # subtract one from one-based functions to convert to zero-based indexing
    subtract_one_from_one_indexed_functions = (
        p.WindowFunction(p.RankBase | p.NTile)
        | p.StringFind
        | p.FindInSet
        | p.ArrayPosition
    ) >> c.Subtract(_, 1)

    add_one_to_nth_value_input = p.NthValue >> _.copy(nth=c.Add(_.nth, 1))

    nullify_empty_string_results = (p.ExtractURLField | p.DayOfWeekName) >> c.NullIf(
        _, ""
    )

    op = op.replace(
        replace_literals
        | replace_in_column_with_table_array_view
        | replace_empty_in_values_with_false
        | subtract_one_from_one_indexed_functions
        | add_one_to_nth_value_input
        | nullify_empty_string_results
        | rewrite_fillna
        | rewrite_dropna
        | rewrite_sample
    )
    # apply translate rules in topological order
    node = op.map(fn)[op]
    return node.this if isinstance(node, sg.exp.Subquery) else node
