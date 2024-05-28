"""Operations for working with maps."""

from __future__ import annotations

from public import public

import ibis.expr.datatypes as dt
import ibis.expr.rules as rlz
from ibis.common.annotations import attribute
from ibis.expr.operations.core import Unary, Value


@public
class Map(Value):
    """Construct a map."""

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
    """Compute the number of unique keys in a map."""

    arg: Value[dt.Map]
    dtype = dt.int64


@public
class MapGet(Value):
    """Get a value from a map by key."""

    arg: Value[dt.Map]
    key: Value
    default: Value = None

    shape = rlz.shape_like("args")

    @attribute
    def dtype(self):
        return dt.higher_precedence(self.default.dtype, self.arg.dtype.value_type)


@public
class MapContains(Value):
    """Check if a map contains a key."""

    arg: Value[dt.Map]
    key: Value

    shape = rlz.shape_like("args")
    dtype = dt.bool


@public
class MapKeys(Unary):
    """Get the keys of a map as an array."""

    arg: Value[dt.Map]

    @attribute
    def dtype(self):
        return dt.Array(self.arg.dtype.key_type)


@public
class MapValues(Unary):
    """Get the values of a map as an array."""

    arg: Value[dt.Map]

    @attribute
    def dtype(self):
        return dt.Array(self.arg.dtype.value_type)


@public
class MapMerge(Value):
    """Combine two maps into one.

    If a key is present in both maps, the value from the first is kept.
    """

    left: Value[dt.Map]
    right: Value[dt.Map]

    shape = rlz.shape_like("args")
    dtype = rlz.dtype_like("args")


public(MapValueForKey=MapGet, MapValueOrDefaultForKey=MapGet, MapConcat=MapMerge)
