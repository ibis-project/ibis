"""Sort key operations."""

from __future__ import annotations

from public import public

import ibis.expr.rules as rlz
from ibis.expr.operations.core import Value


@public
class SortKey(Value):
    """A sort operation."""

    expr = rlz.any
    ascending = rlz.optional(rlz.bool_, default=True)

    output_dtype = rlz.dtype_like("expr")
    output_shape = rlz.shape_like("expr")

    @property
    def name(self) -> str:
        return self.expr.name

    @property
    def descending(self) -> bool:
        return not self.ascending
