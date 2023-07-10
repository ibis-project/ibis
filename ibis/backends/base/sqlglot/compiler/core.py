"""ClickHouse ibis expression to sqlglot compiler.

The compiler is built with a few `singledispatch` functions:

    1. `translate` for table expressions
    1. `translate` for table nodes
    1. `translate_rel`
    1. `translate_val`

## `translate`

### Expression Implementation

The table expression implementation of `translate` is a pass through to the
node implementation.

### Node Implementation

There's a single `ops.Node` implementation for `ops.TableNode`s instances.

This function:

    1. Topologically sorts the expression graph.
    1. Seeds the compilation cache with in-degree-zero table names.
    1. Iterates though nodes with at least one in-degree and places the result
       in the compilation cache. The cache is used to construct `ops.TableNode`
       keyword arguments to the current translation rule.

## `translate_rel`

Translates a table operation given already-translated table inputs.

If a table node needs to translate value expressions, for example, an
`ops.Aggregation` that rule is responsible for calling `translate_val`.

## `translate_val`

Recurses top-down and translates the arguments of the value expression and uses
those as input to construct the output.
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Any

import sqlglot as sg

import ibis.expr.analysis as an
import ibis.expr.operations as ops
import ibis.expr.types as ir
from ibis.common.patterns import Call, _
from ibis.expr.analysis import c, p, x, y

if TYPE_CHECKING:
    from collections.abc import Mapping


a = Call.namespace(an)


def translate(
    op: ops.TableNode, *, params: Mapping[ir.Value, Any], translate_rel, translate_val
) -> sg.exp.Expression:
    """Translate an ibis operation to a sqlglot expression.

    Parameters
    ----------
    op
        An ibis `TableNode`
    params
        A mapping of expressions to concrete values
    translate_rel
        Relation node translator
    translate_val
        Value node translator

    Returns
    -------
    sqlglot.expressions.Expression
        A sqlglot expression
    """

    def _translate_node(node, **kwargs):
        if isinstance(node, ops.Value):
            return translate_val(node, **kwargs)
        assert isinstance(node, ops.TableNode)
        return translate_rel(node, **kwargs)

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
        lambda op, ctx: ops.Literal(value=params[op], dtype=ctx[x])
    )

    # replace the right side of InColumn into a scalar subquery for sql
    # backends
    replace_in_column_with_table_array_view = p.InColumn(..., y) >> _.copy(
        options=c.TableArrayView(
            c.Selection(table=a.find_first_base_table(y), selections=(y,))
        ),
    )

    # replace any checks against an empty right side of the IN operation with
    # `False`
    replace_empty_in_values_with_false = p.InValues(..., ()) >> c.Literal(
        False, dtype="bool"
    )

    # replace `NotExistsSubquery` with `Not(ExistsSubquery)`
    #
    # this allows us to avoid having another rule to negate ExistsSubquery
    replace_notexists_subquery_with_not_exists = p.NotExistsSubquery(x, y) >> c.Not(
        c.ExistsSubquery(x, y)
    )

    # clickhouse-specific rewrite to turn notany/notall into equivalent
    # already-defined operations
    replace_notany_with_min_not = p.NotAny(x, where=y) >> c.Min(c.Not(x), where=y)
    replace_notall_with_max_not = p.NotAll(x, where=y) >> c.Max(c.Not(x), where=y)

    # add an ORDER BY clause to rank window functions that don't have one
    add_order_by_to_empty_window_functions = p.WindowFunction(
        p.PercentRank(x) | p.RankBase(x) | p.CumeDist(x) | p.NTile(x),
        p.WindowFrame(..., order_by=()) >> _.copy(order_by=(x,)),
    )

    op = op.replace(
        replace_literals
        | replace_in_column_with_table_array_view
        | replace_empty_in_values_with_false
        | replace_notexists_subquery_with_not_exists
        | replace_notany_with_min_not
        | replace_notall_with_max_not
        | add_order_by_to_empty_window_functions
    )
    # apply translate rules in topological order
    results = op.map(fn, filter=(ops.TableNode, ops.Value))
    node = results[op]
    return node.this if isinstance(node, sg.exp.Subquery) else node
