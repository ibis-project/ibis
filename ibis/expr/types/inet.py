from __future__ import annotations

from public import public

from ibis.expr import datatypes as dt
from ibis.expr.types.generic import Column, Scalar, Value


@public
class MACADDRValue(Value):
    __dtype__ = dt.macaddr


@public
class MACADDRScalar(Scalar, MACADDRValue):
    pass


@public
class MACADDRColumn(Column, MACADDRValue):
    pass


@public
class INETValue(Value):
    __dtype__ = dt.inet


@public
class INETScalar(Scalar, INETValue):
    pass


@public
class INETColumn(Column, INETValue):
    pass
