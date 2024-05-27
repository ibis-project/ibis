"""Operations for working with JSON data."""

from __future__ import annotations

from public import public

import ibis.expr.datatypes as dt
import ibis.expr.rules as rlz
from ibis.expr.operations import Value


@public
class JSONGetItem(Value):
    """Get a value from a JSON object or array."""

    arg: Value[dt.JSON]
    index: Value[dt.String | dt.Integer]

    dtype = dt.json
    shape = rlz.shape_like("args")


@public
class ToJSONArray(Value):
    """Convert a value to an array of JSON objects."""

    arg: Value[dt.JSON]

    dtype = dt.Array(dt.json)
    shape = rlz.shape_like("arg")


@public
class ToJSONMap(Value):
    """Convert a value to a map of string to JSON."""

    arg: Value[dt.JSON]

    dtype = dt.Map(dt.string, dt.json)
    shape = rlz.shape_like("arg")


@public
class UnwrapJSONString(Value):
    """Unwrap a JSON string into an engine-native string."""

    arg: Value[dt.JSON]

    dtype = dt.string
    shape = rlz.shape_like("arg")


@public
class UnwrapJSONInt64(Value):
    """Unwrap a JSON number into an engine-native int64."""

    arg: Value[dt.JSON]

    dtype = dt.int64
    shape = rlz.shape_like("arg")


@public
class UnwrapJSONFloat64(Value):
    """Unwrap a JSON number into an engine-native float64."""

    arg: Value[dt.JSON]

    dtype = dt.float64
    shape = rlz.shape_like("arg")


@public
class UnwrapJSONBoolean(Value):
    """Unwrap a JSON bool into an engine-native bool."""

    arg: Value[dt.JSON]

    dtype = dt.boolean
    shape = rlz.shape_like("arg")
