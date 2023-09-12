from __future__ import annotations

from public import public

from ibis.expr.types.generic import Column, Scalar, Value


@public
class MACADDRValue(Value):
    pass


@public
class MACADDRScalar(Scalar, MACADDRValue):
    pass


@public
class MACADDRColumn(Column, MACADDRValue):
    pass


@public
class INETValue(Value):
    pass


@public
class INETScalar(Scalar, INETValue):
    pass


@public
class INETColumn(Column, INETValue):
    pass
