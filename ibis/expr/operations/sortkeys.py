from public import public

import ibis.expr.rules as rlz
from ibis.expr.operations.core import Value


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
        what = parent.to_expr()._ensure_expr(self.what)
        return SortKey(what, ascending=self.ascending)
