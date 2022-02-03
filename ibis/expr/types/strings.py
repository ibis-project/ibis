from public import public

from .generic import AnyColumn, AnyScalar, AnyValue


@public
class StringValue(AnyValue):
    pass  # noqa: E701,E302


@public
class StringScalar(AnyScalar, StringValue):
    pass  # noqa: E701,E302


@public
class StringColumn(AnyColumn, StringValue):
    pass  # noqa: E701,E302
