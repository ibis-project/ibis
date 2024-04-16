from __future__ import annotations

from public import public

import ibis.expr.datatypes as dt
import ibis.expr.rules as rlz
from ibis.expr.operations import Value


@public
class JSONGetItem(Value):
    arg: Value[dt.JSON]
    index: Value[dt.String | dt.Integer]

    dtype = dt.json
    shape = rlz.shape_like("args")


@public
class ToJSONArray(Value):
    arg: Value[dt.JSON]

    dtype = dt.Array(dt.json)
    shape = rlz.shape_like("arg")


@public
class ToJSONMap(Value):
    arg: Value[dt.JSON]

    dtype = dt.Map(dt.string, dt.json)
    shape = rlz.shape_like("arg")


@public
class UnwrapJSONString(Value):
    arg: Value[dt.JSON]

    dtype = dt.string
    shape = rlz.shape_like("arg")


@public
class UnwrapJSONInt64(Value):
    arg: Value[dt.JSON]

    dtype = dt.int64
    shape = rlz.shape_like("arg")


@public
class UnwrapJSONFloat64(Value):
    arg: Value[dt.JSON]

    dtype = dt.float64
    shape = rlz.shape_like("arg")


@public
class UnwrapJSONBoolean(Value):
    arg: Value[dt.JSON]

    dtype = dt.boolean
    shape = rlz.shape_like("arg")
