from __future__ import annotations

from public import public

from ibis.expr.types.generic import Value


# TODO(kszucs): we don't need to separate SortExpr objects from regular value
# expressions
@public
class SortExpr(Value):
    pass
