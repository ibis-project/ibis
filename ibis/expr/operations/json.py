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
