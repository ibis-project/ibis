from public import public

from ibis.expr.types.binary import BinaryColumn, BinaryScalar, BinaryValue
from ibis.expr.types.strings import StringColumn, StringScalar, StringValue


@public
class JSONValue(StringValue):
    pass  # noqa: E701,E302


@public
class JSONScalar(StringScalar, JSONValue):
    pass  # noqa: E701,E302


@public
class JSONColumn(StringColumn, JSONValue):
    pass  # noqa: E701,E302


@public
class JSONBValue(BinaryValue):
    pass  # noqa: E701,E302


@public
class JSONBScalar(BinaryScalar, JSONBValue):
    pass  # noqa: E701,E302


@public
class JSONBColumn(BinaryColumn, JSONBValue):
    pass  # noqa: E701,E302
