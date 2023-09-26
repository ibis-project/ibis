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

from typing import TYPE_CHECKING, Any

import sqlglot as sg

import ibis.expr.analysis as an
import ibis.expr.operations as ops
import ibis.expr.types as ir
from ibis.common.patterns import Call
from ibis.expr.analysis import c, p, x, y

if TYPE_CHECKING:
    from collections.abc import Mapping


a = Call.namespace(an)


def translate(
    op: ops.TableNode,
    *,
    params: Mapping[ir.Value, Any],
    translate_rel,
    translate_val,
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

    def _translate_node(node, *args, **kwargs):
        if isinstance(node, ops.Value):
            return translate_val(node, *args, **kwargs)
        assert isinstance(node, ops.TableNode)
        return translate_rel(node, *args, **kwargs)

    alias_index = 0
    aliases = {}

    def fn(node, _, **kwargs):
        nonlocal alias_index

        result = _translate_node(node, aliases=aliases, **kwargs)

        if not isinstance(node, ops.TableNode):
            return result

        # don't alias the root node
        if node is not op:
            aliases[node] = f"t{alias_index:d}"
            alias_index += 1

        if alias := aliases.get(node):
            try:
                return result.subquery(alias=alias)
            except AttributeError:
                return sg.alias(result, alias=alias)
        else:
            return result

    # substitute parameters immediately to avoid having to define a
    # ScalarParameter translation rule
    #
    # this lets us avoid threading `params` through every `translate_val` call
    # only to be used in the one place it would be needed: the ScalarParameter
    # `translate_val` rule
    params = {param.op(): value for param, value in params.items()}
    replace_literals = p.ScalarParameter >> (
        lambda op, _: ops.Literal(value=params[op], dtype=op.dtype)
    )

    # rewrite cumulative functions to window functions, so that we don't have
    # to think about handling them in the compiler, we need only compile window
    # functions
    replace_cumulative_ops = p.WindowFunction(
        x @ p.Cumulative, y
    ) >> a.cumulative_to_window(x, y)

    # replace the right side of InColumn into a scalar subquery for sql
    # backends
    replace_in_column_with_table_array_view = p.InColumn >> (
        lambda op, _: op.__class__(
            op.value,
            ops.TableArrayView(
                ops.Selection(
                    table=an.find_first_base_table(op.options), selections=(op.options,)
                )
            ),
        )
    )

    # replace any checks against an empty right side of the IN operation with
    # `False`
    replace_empty_in_values_with_false = p.InValues(x, ()) >> c.Literal(
        False, dtype="bool"
    )

    replace_notexists_subquery_with_not_exists = p.NotExistsSubquery(x, y) >> c.Not(
        c.ExistsSubquery(x, y)
    )

    add_order_by_to_window_funcs = p.WindowFunction(
        p.PercentRank(x) | p.RankBase(x) | p.CumeDist(x),
        (
            p.WindowFrame(..., order_by=())
            >> (lambda op, ctx: op.copy(order_by=(ctx[x],)))
        ),
    )

    op = op.replace(
        replace_literals
        | replace_cumulative_ops
        | replace_in_column_with_table_array_view
        | replace_empty_in_values_with_false
        | replace_notexists_subquery_with_not_exists
        | add_order_by_to_window_funcs
    )
    # apply translate rules in topological order
    results = op.map(fn, filter=(ops.TableNode, ops.Value))
    node = results[op]
    return node.this if isinstance(node, sg.exp.Subquery) else node
