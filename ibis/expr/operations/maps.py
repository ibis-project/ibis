from __future__ import annotations

from public import public

import ibis.expr.datatypes as dt
import ibis.expr.rules as rlz
from ibis.common.annotations import attribute
from ibis.expr.operations.core import Unary, Value
from ibis.expr.types.generic import null


@public
class Map(Value):
    keys = rlz.array
    values = rlz.array

    output_shape = rlz.shape_like("args")

    @attribute.default
    def output_dtype(self):
        return dt.Map(
            self.keys.output_dtype.value_type,
            self.values.output_dtype.value_type,
        )


@public
class MapLength(Unary):
    arg = rlz.mapping
    output_dtype = dt.int64


@public
class MapGet(Value):
    arg = rlz.mapping
    key = rlz.one_of([rlz.string, rlz.integer])
    default = rlz.optional(rlz.any, default=null())

    output_shape = rlz.shape_like("args")

    @attribute.default
    def output_dtype(self):
        return dt.higher_precedence(
            self.default.output_dtype, self.arg.output_dtype.value_type
        )


@public
class MapContains(Value):
    arg = rlz.mapping
    key = rlz.one_of([rlz.string, rlz.integer])

    output_shape = rlz.shape_like("args")
    output_dtype = dt.bool


@public
class MapKeys(Unary):
    arg = rlz.mapping

    @attribute.default
    def output_dtype(self):
        return dt.Array(self.arg.output_dtype.key_type)


@public
class MapValues(Unary):
    arg = rlz.mapping

    @attribute.default
    def output_dtype(self):
        return dt.Array(self.arg.output_dtype.value_type)


@public
class MapMerge(Value):
    left = rlz.mapping
    right = rlz.mapping

    output_shape = rlz.shape_like("args")
    output_dtype = rlz.dtype_like("args")


public(MapValueForKey=MapGet, MapValueOrDefaultForKey=MapGet, MapConcat=MapMerge)
