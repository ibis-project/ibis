from __future__ import annotations

from public import public

import ibis.expr.datatypes as dt
import ibis.expr.rules as rlz
from ibis.expr.operations import Value


@public
class JSONGetItem(Value):
    arg = rlz.json
    index = rlz.one_of((rlz.string, rlz.integer))

    output_dtype = dt.json
    output_shape = rlz.shape_like("args")


@public
class ToJSONArray(Value):
    arg = rlz.json

    output_dtype = dt.Array(dt.json)
    output_shape = rlz.shape_like("arg")


@public
class ToJSONMap(Value):
    arg = rlz.json

    output_dtype = dt.Map(dt.string, dt.json)
    output_shape = rlz.shape_like("arg")
