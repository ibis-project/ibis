from __future__ import annotations

from public import public

from ibis.expr.types.strings import StringColumn, StringScalar, StringValue


@public
class MACADDRValue(StringValue):
    pass


@public
class MACADDRScalar(StringScalar, MACADDRValue):
    pass


@public
class MACADDRColumn(StringColumn, MACADDRValue):
    pass


@public
class INETValue(StringValue):
    pass


@public
class INETScalar(StringScalar, INETValue):
    pass


@public
class INETColumn(StringColumn, INETValue):
    pass
