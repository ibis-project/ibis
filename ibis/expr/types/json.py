"""JSON value operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from public import public

from ibis.expr.types import Column, Scalar, Value

if TYPE_CHECKING:
    import ibis.expr.types as ir


@public
class JSONValue(Value):
    def __getitem__(
        self, key: str | int | ir.StringValue | ir.IntegerValue
    ) -> JSONValue:
        import ibis.expr.operations as ops

        return ops.JSONGetItem(self, key).to_expr()


@public
class JSONScalar(Scalar, JSONValue):
    pass


@public
class JSONColumn(Column, JSONValue):
    pass
