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

    output_shape = rlz.shape_like("args")

    @attribute.default
    def output_dtype(self):
        return dt.Map(
            self.keys.output_dtype.value_type,
            self.values.output_dtype.value_type,
        )


@public
class MapLength(Unary):
    arg: Value[dt.Map]
    output_dtype = dt.int64


@public
class MapGet(Value):
    arg: Value[dt.Map]
    key: Value[dt.String | dt.Integer]
    default: Value = None

    output_shape = rlz.shape_like("args")

    @attribute.default
    def output_dtype(self):
        return dt.higher_precedence(
            self.default.output_dtype, self.arg.output_dtype.value_type
        )


@public
class MapContains(Value):
    arg: Value[dt.Map]
    key: Value[dt.String | dt.Integer]

    output_shape = rlz.shape_like("args")
    output_dtype = dt.bool


@public
class MapKeys(Unary):
    arg: Value[dt.Map]

    @attribute.default
    def output_dtype(self):
        return dt.Array(self.arg.output_dtype.key_type)


@public
class MapValues(Unary):
    arg: Value[dt.Map]

    @attribute.default
    def output_dtype(self):
        return dt.Array(self.arg.output_dtype.value_type)


@public
class MapMerge(Value):
    left: Value[dt.Map]
    right: Value[dt.Map]

    output_shape = rlz.shape_like("args")
    output_dtype = rlz.dtype_like("args")


public(MapValueForKey=MapGet, MapValueOrDefaultForKey=MapGet, MapConcat=MapMerge)
