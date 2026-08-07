from __future__ import annotations

from typing import TYPE_CHECKING

import ibis.expr.operations as ops
from ibis.common.exceptions import IbisError

if TYPE_CHECKING:
    from collections.abc import Iterable

    import ibis.expr.types as ir
    from ibis.backends import BaseBackend


def find_backends(nodish: ir.Expr | ops.Node, /) -> tuple[list[BaseBackend], bool]:
    """Return the possible backends for an expression.

    Returns
    -------
    tuple[list[BaseBackend], bool]
        A list of the backends found, and a boolean indicating whether there
        are any unbound tables in the expression.
    """
    node = ensure_node(nodish)
    backends = set()
    has_unbound = False
    node_types = (ops.UnboundTable, ops.DatabaseTable, ops.SQLQueryResult)
    for table in node.find(node_types):
        if isinstance(table, ops.UnboundTable):
            has_unbound = True
        else:
            backends.add(table.source)

    return list(backends), has_unbound


def find_backend(
    nodes: ir.Expr | ops.Node | Iterable[ir.Expr | ops.Node], /
) -> BaseBackend | None:
    """Find the backend attached to some expressions, if any.

    Parameters
    ----------
    nodes : Expr or Node, or Iterable[Expr or Node]
        The expressions to find the backend for.

    Returns
    -------
    BaseBackend | None
        A backend that is attached to one of the expressions, or `None` if no backend
        is found.

    Raises
    ------
    IbisError
        If multiple backends are found.
    """
    import ibis.expr.types as ir

    if isinstance(nodes, ir.Expr):
        n = [nodes.op()]
    elif isinstance(nodes, ops.Node):
        n = [nodes]
    else:
        n = [ensure_node(node) for node in nodes]

    raw_backends = {_find_backend(node) for node in n}
    backends = {b for b in raw_backends if b is not None}
    if not backends:
        return None
    if len(backends) > 1:
        raise IbisError(
            f"Cannot determine backend from values with multiple backends: {backends}"
        )
    result = next(iter(backends))
    return result


def _find_backend(node: ops.Node, /) -> BaseBackend | None:
    backends, has_unbound = find_backends(node)
    if not backends:
        if has_unbound:
            raise IbisError(
                "Expression contains unbound tables and therefore cannot "
                "be executed. Use `<backend>.execute(expr)` to execute "
                "against an explicit backend, or rebuild the expression "
                "using bound tables instead."
            )
        return None
    if len(backends) > 1:
        raise IbisError("Multiple backends found for this expression")
    return backends[0]


def ensure_node(raw: ir.Expr | ops.Node) -> ops.Node:
    if isinstance(raw, ops.Node):
        return raw
    import ibis.expr.types as ir

    if isinstance(raw, ir.Expr):
        return raw.op()
    raise TypeError(f"Could not convert object {raw} of type {type(raw)} to Node")
