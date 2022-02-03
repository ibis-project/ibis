from __future__ import annotations

from public import public

from .generic import Expr


@public
class SortExpr(Expr):
    def _type_display(self) -> str:
        return 'array-sort'

    def get_name(self) -> str | None:
        return self.op().resolve_name()
