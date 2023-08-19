from __future__ import annotations

from public import public

import ibis.expr.datatypes as dt
import ibis.expr.rules as rlz
from ibis.common.annotations import attribute
from ibis.expr.operations.core import Unary, Value


@public
class Map(Value):
    keys: Value[dt.Array]
    values: Value[dt.Array]

    shape = rlz.shape_like("args")

    @attribute
    def dtype(self):
        return dt.Map(
            self.keys.dtype.value_type,
            self.values.dtype.value_type,
        )


@public
class MapLength(Unary):
    arg: Value[dt.Map]
    dtype = dt.int64


@public
class MapGet(Value):
    arg: Value[dt.Map]
    key: Value[dt.String | dt.Integer]
    default: Value = None

    shape = rlz.shape_like("args")

    @attribute
    def dtype(self):
        return dt.higher_precedence(self.default.dtype, self.arg.dtype.value_type)


@public
class MapContains(Value):
    arg: Value[dt.Map]
    key: Value[dt.String | dt.Integer]

    shape = rlz.shape_like("args")
    dtype = dt.bool


@public
class MapKeys(Unary):
    arg: Value[dt.Map]

    @attribute
    def dtype(self):
        return dt.Array(self.arg.dtype.key_type)


@public
class MapValues(Unary):
    arg: Value[dt.Map]

    @attribute
    def dtype(self):
        return dt.Array(self.arg.dtype.value_type)


@public
class MapMerge(Value):
    left: Value[dt.Map]
    right: Value[dt.Map]

    shape = rlz.shape_like("args")
    dtype = rlz.dtype_like("args")


public(MapValueForKey=MapGet, MapValueOrDefaultForKey=MapGet, MapConcat=MapMerge)
