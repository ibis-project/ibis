from __future__ import annotations

from public import public

from .generic import Expr


@public
class SortExpr(Expr):
    def get_name(self) -> str | None:
        return self.op().resolve_name()
