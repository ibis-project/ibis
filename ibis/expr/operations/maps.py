from public import public

import ibis.expr.datatypes as dt
import ibis.expr.rules as rlz
from ibis.common.validators import immutable_property
from ibis.expr.operations.core import Unary, Value


@public
class MapLength(Unary):
    arg = rlz.mapping
    output_dtype = dt.int64


@public
class MapGet(Value):
    arg = rlz.mapping
    key = rlz.one_of([rlz.string, rlz.integer])

    output_shape = rlz.shape_like("args")

    @immutable_property
    def output_dtype(self):
        return self.arg.output_dtype.value_type


@public
class MapGetOr(MapGet):
    default = rlz.any

    @immutable_property
    def output_dtype(self):
        return dt.higher_precedence(
            self.default.output_dtype, self.arg.output_dtype.value_type
        )


@public
class MapKeys(Unary):
    arg = rlz.mapping

    @immutable_property
    def output_dtype(self):
        return dt.Array(self.arg.output_dtype.key_type)


@public
class MapValues(Unary):
    arg = rlz.mapping

    @immutable_property
    def output_dtype(self):
        return dt.Array(self.arg.output_dtype.value_type)


@public
class MapConcat(Value):
    left = rlz.mapping
    right = rlz.mapping

    output_shape = rlz.shape_like("args")
    output_dtype = rlz.dtype_like("args")


public(MapValueForKey=MapGet, MapValueOrDefaultForKey=MapGetOr)
