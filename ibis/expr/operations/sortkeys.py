from public import public

from ibis import util
from ibis.common import exceptions as com
from ibis.expr import rules as rlz
from ibis.expr import types as ir
from ibis.expr.operations.core import Node


def _to_sort_key(key, *, table=None):
    if isinstance(key, DeferredSortKey):
        if table is None:
            raise com.IbisTypeError(
                "cannot resolve DeferredSortKey with table=None"
            )
        key = key.resolve(table)

    if isinstance(key, ir.SortExpr):
        return key

    if isinstance(key, (tuple, list)):
        key, sort_order = key
    else:
        sort_order = True

    if not isinstance(key, ir.Expr):
        if table is None:
            raise com.IbisTypeError("cannot resolve key with table=None")
        key = table._ensure_expr(key)
        if isinstance(key, (ir.SortExpr, DeferredSortKey)):
            return _to_sort_key(key, table=table)

    if isinstance(sort_order, str):
        if sort_order.lower() in ('desc', 'descending'):
            sort_order = False
        elif not isinstance(sort_order, bool):
            sort_order = bool(sort_order)

    return SortKey(key, ascending=sort_order).to_expr()


def _maybe_convert_sort_keys(tables, exprs):
    exprs = util.promote_list(exprs)
    keys = exprs[:]
    for i, key in enumerate(exprs):
        for table in reversed(tables):
            try:
                sort_key = _to_sort_key(key, table=table)
            except Exception:
                continue
            else:
                keys[i] = sort_key
                break
    return keys


@public
class SortKey(Node):
    expr = rlz.any
    ascending = rlz.optional(
        rlz.map_to(
            {
                True: True,
                False: False,
                1: True,
                0: False,
            },
        ),
        default=True,
    )

    output_type = ir.SortExpr

    def root_tables(self):
        return self.expr.op().root_tables()

    def resolve_name(self):
        return self.expr.get_name()


@public
class DeferredSortKey:
    def __init__(self, what, ascending=True):
        self.what = what
        self.ascending = ascending

    def resolve(self, parent):
        what = parent._ensure_expr(self.what)
        return SortKey(what, ascending=self.ascending).to_expr()
