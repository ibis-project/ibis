from __future__ import annotations

from typing import TYPE_CHECKING

from public import public

if TYPE_CHECKING:
    import ibis.expr.datatypes as dt

from ibis.expr.types.generic import Expr


@public
class SortExpr(Expr):
    def type(self) -> dt.DataType:
        return self.op().expr.type()
