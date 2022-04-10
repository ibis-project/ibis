from public import public

from ibis.expr.types.strings import StringColumn, StringScalar, StringValue


@public
class UUIDValue(StringValue):
    pass  # noqa: E701,E302


@public
class UUIDScalar(StringScalar, UUIDValue):
    pass  # noqa: E701,E302


@public
class UUIDColumn(StringColumn, UUIDValue):
    pass  # noqa: E701,E302
