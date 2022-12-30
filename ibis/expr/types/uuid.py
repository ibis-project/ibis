from __future__ import annotations

from public import public

from ibis.expr.types.strings import StringColumn, StringScalar, StringValue


@public
class UUIDValue(StringValue):
    pass


@public
class UUIDScalar(StringScalar, UUIDValue):
    pass


@public
class UUIDColumn(StringColumn, UUIDValue):
    pass
