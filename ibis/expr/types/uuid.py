from __future__ import annotations

from typing import TYPE_CHECKING

from public import public

from ibis.expr.types.generic import Column, Scalar, Value

if TYPE_CHECKING:
    from ibis.common.deferred import Deferred
    from ibis.expr import types as ir
    from ibis.expr.datatypes.value import InferrableToString, InferrableToUUID


@public
class UUIDValue(Value):
    def __eq__(
        self,
        other: InferrableToUUID
        | UUIDValue
        | InferrableToString
        | ir.StringValue
        | Deferred,
    ) -> ir.BooleanValue:
        return super().__eq__(other)

    def __ne__(
        self,
        other: InferrableToUUID
        | UUIDValue
        | InferrableToString
        | ir.StringValue
        | Deferred,
    ) -> ir.BooleanValue:
        return super().__ne__(other)

    def __ge__(
        self,
        other: InferrableToUUID
        | UUIDValue
        | InferrableToString
        | ir.StringValue
        | Deferred,
    ) -> ir.BooleanValue:
        return super().__ge__(other)

    def __gt__(
        self,
        other: InferrableToUUID
        | UUIDValue
        | InferrableToString
        | ir.StringValue
        | Deferred,
    ) -> ir.BooleanValue:
        return super().__gt__(other)

    def __le__(
        self,
        other: InferrableToUUID
        | UUIDValue
        | InferrableToString
        | ir.StringValue
        | Deferred,
    ) -> ir.BooleanValue:
        return super().__le__(other)

    def __lt__(
        self,
        other: InferrableToUUID
        | UUIDValue
        | InferrableToString
        | ir.StringValue
        | Deferred,
    ) -> ir.BooleanValue:
        return super().__lt__(other)


@public
class UUIDScalar(Scalar, UUIDValue):
    pass


@public
class UUIDColumn(Column, UUIDValue):
    pass
