from public import public

from ibis.expr.types.strings import StringColumn, StringScalar, StringValue


@public
class MACADDRValue(StringValue):
    pass  # noqa: E701,E302


@public
class MACADDRScalar(StringScalar, MACADDRValue):
    pass  # noqa: E701,E302


@public
class MACADDRColumn(StringColumn, MACADDRValue):
    pass  # noqa: E701,E302


@public
class INETValue(StringValue):
    pass  # noqa: E701,E302


@public
class INETScalar(StringScalar, INETValue):
    pass  # noqa: E701,E302


@public
class INETColumn(StringColumn, INETValue):
    pass  # noqa: E701,E302
