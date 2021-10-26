from public import public

from ... import util
from ...common import exceptions as com
from .. import rules as rlz
from .. import types as ir
from ..signature import Argument as Arg
from .core import Node, _safe_repr


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
        step = -1 if isinstance(key, (str, DeferredSortKey)) else 1
        for table in tables[::step]:
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
    expr = Arg(rlz.column(rlz.any))
    ascending = Arg(
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

    def __repr__(self):
        # Temporary
        rows = [
            'Sort key:',
            f'  ascending: {self.ascending!s}',
            util.indent(_safe_repr(self.expr), 2),
        ]
        return '\n'.join(rows)

    def output_type(self):
        return ir.SortExpr

    def root_tables(self):
        return self.expr.op().root_tables()

    def equals(self, other, cache=None):
        return (
            isinstance(other, SortKey)
            and self.expr.equals(other.expr, cache=cache)
            and self.ascending == other.ascending
        )

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
