from public import public

import ibis.expr.operations as ops
import ibis.expr.rules as rlz
from ibis import util
from ibis.common import exceptions as com
from ibis.expr.operations.core import Value


# TODO(kszucs): rewrite to both receive operations and return with operations
def _to_sort_key(key, *, table=None):
    import ibis.expr.types as ir

    if isinstance(key, DeferredSortKey):
        if table is None:
            raise com.IbisTypeError(
                "cannot resolve DeferredSortKey with table=None"
            )
        key = key.resolve(table)

    # TODO(kszucs): refactor to only work with operation classes
    if isinstance(key, ir.SortExpr):
        return key

    if isinstance(key, ops.SortKey):
        return key.to_expr()

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


# TODO(kszucs): rewrite to both receive operations and return with operations
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

    return [k.op() for k in keys]


@public
class SortKey(Value):
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

    output_shape = rlz.Shape.COLUMNAR

    def resolve_name(self):
        return self.expr.resolve_name()

    @property
    def output_dtype(self):
        return self.expr.output_dtype

    # TODO(kszucs): should either return with a regular value expression or
    # shoulnd't be a subclass of ops.Value
    def to_expr(self):
        import ibis.expr.types as ir

        return ir.SortExpr(self)


@public
class DeferredSortKey:
    def __init__(self, what, ascending=True):
        self.what = what
        self.ascending = ascending

    def resolve(self, parent):
        what = parent._ensure_expr(self.what)
        return SortKey(what, ascending=self.ascending).to_expr()
