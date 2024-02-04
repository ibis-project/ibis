from __future__ import annotations

import ibis.common.graph as g
import ibis.expr.operations as ops
from ibis.common.deferred import deferred, var
from ibis.common.patterns import pattern
from ibis.util import Namespace

p = Namespace(pattern, module=ops)
c = Namespace(deferred, module=ops)

x = var("x")
y = var("y")


# TODO(kszucs): should be removed
def find_first_base_table(node):
    def predicate(node):
        if isinstance(node, ops.Relation):
            return g.halt, node
        else:
            return g.proceed, None

    try:
        return next(g.traverse(predicate, node))
    except StopIteration:
        return None


def flatten_predicates(node):
    """Yield the expressions corresponding to the `And` nodes of a predicate.

    Examples
    --------
    >>> import ibis
    >>> t = ibis.table([("a", "int64"), ("b", "string")], name="t")
    >>> filt = (t.a == 1) & (t.b == "foo")
    >>> predicates = flatten_predicates(filt.op())
    >>> len(predicates)
    2
    >>> predicates[0].to_expr().name("left")
    r0 := UnboundTable: t
      a int64
      b string
    left: r0.a == 1
    >>> predicates[1].to_expr().name("right")
    r0 := UnboundTable: t
      a int64
      b string
    right: r0.b == 'foo'

    """

    def predicate(node):
        if isinstance(node, ops.And):
            return g.proceed, None
        else:
            return g.halt, node

    return list(g.traverse(predicate, node))
