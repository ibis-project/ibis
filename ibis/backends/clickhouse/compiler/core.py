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

from typing import Any, Mapping

import sqlglot as sg

import ibis.expr.operations as ops
import ibis.expr.types as ir
from ibis.backends.clickhouse.compiler.relations import translate_rel


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
    params = {param.op(): value for param, value in params.items()}

    alias_index = 0
    aliases = {}

    def fn(node, cache, params=params, **kwargs):
        nonlocal alias_index

        # don't alias the root node
        if node is not op:
            aliases[node] = f"t{alias_index:d}"
            alias_index += 1

        raw_rel = translate_rel(
            node, aliases=aliases, params=params, cache=cache, **kwargs
        )

        if alias := aliases.get(node):
            try:
                return raw_rel.subquery(alias)
            except AttributeError:
                return sg.alias(raw_rel, alias)
        else:
            return raw_rel

    results = op.map(fn, filter=ops.TableNode)
    node = results[op]
    return node.this if isinstance(node, sg.exp.Subquery) else node
