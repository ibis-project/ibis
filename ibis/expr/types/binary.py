from public import public

from .generic import AnyColumn, AnyScalar, AnyValue


@public
class BinaryValue(AnyValue):
    pass  # noqa: E701,E302


@public
class BinaryScalar(AnyScalar, BinaryValue):
    pass  # noqa: E701,E302


@public
class BinaryColumn(AnyColumn, BinaryValue):
    pass  # noqa: E701,E302
