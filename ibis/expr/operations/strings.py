from __future__ import annotations

from typing import Optional

from public import public

import ibis.expr.datashape as ds
import ibis.expr.datatypes as dt
import ibis.expr.rules as rlz
from ibis.common.annotations import attribute
from ibis.common.typing import VarTuple  # noqa: TCH001
from ibis.expr.operations.core import Unary, Value


@public
class StringUnary(Unary):
    arg: Value[dt.String]

    output_dtype = dt.string


@public
class Uppercase(StringUnary):
    pass


@public
class Lowercase(StringUnary):
    pass


@public
class Reverse(StringUnary):
    pass


@public
class Strip(StringUnary):
    pass


@public
class LStrip(StringUnary):
    pass


@public
class RStrip(StringUnary):
    pass


@public
class Capitalize(StringUnary):
    pass


@public
class Substring(Value):
    arg: Value[dt.String]
    start: Value[dt.Integer]
    length: Optional[Value[dt.Integer]] = None

    output_dtype = dt.string
    output_shape = rlz.shape_like('arg')


@public
class StrRight(Value):
    arg: Value[dt.String]
    nchars: Value[dt.Integer]

    output_shape = rlz.shape_like("arg")
    output_dtype = dt.string


@public
class Repeat(Value):
    arg: Value[dt.String]
    times: Value[dt.Integer]

    output_shape = rlz.shape_like("arg")
    output_dtype = dt.string


@public
class StringFind(Value):
    arg: Value[dt.String]
    substr: Value[dt.String]
    start: Optional[Value[dt.Integer]] = None
    end: Optional[Value[dt.Integer]] = None

    output_shape = rlz.shape_like("arg")
    output_dtype = dt.int64


@public
class Translate(Value):
    arg: Value[dt.String]
    from_str: Value[dt.String]
    to_str: Value[dt.String]

    output_shape = rlz.shape_like("arg")
    output_dtype = dt.string


@public
class LPad(Value):
    arg: Value[dt.String]
    length: Value[dt.Integer]
    pad: Optional[Value[dt.String]] = None

    output_shape = rlz.shape_like("arg")
    output_dtype = dt.string


@public
class RPad(Value):
    arg: Value[dt.String]
    length: Value[dt.Integer]
    pad: Optional[Value[dt.String]] = None

    output_shape = rlz.shape_like("arg")
    output_dtype = dt.string


@public
class FindInSet(Value):
    needle: Value[dt.String]
    values: VarTuple[Value[dt.String]]

    output_shape = rlz.shape_like("needle")
    output_dtype = dt.int64


@public
class StringJoin(Value):
    sep: Value[dt.String]
    arg: VarTuple[Value[dt.String]]

    output_dtype = dt.string

    @attribute.default
    def output_shape(self):
        return rlz.highest_precedence_shape(self.arg)


@public
class ArrayStringJoin(Value):
    sep: Value[dt.String]
    arg: Value[dt.Array[dt.String]]

    output_dtype = dt.string
    output_shape = rlz.shape_like("args")


@public
class StartsWith(Value):
    arg: Value[dt.String]
    start: Value[dt.String, ds.Scalar]

    output_dtype = dt.boolean
    output_shape = rlz.shape_like("arg")


@public
class EndsWith(Value):
    arg: Value[dt.String]
    end: Value[dt.String, ds.Scalar]

    output_dtype = dt.boolean
    output_shape = rlz.shape_like("arg")


@public
class FuzzySearch(Value):
    arg: Value[dt.String]
    pattern: Value[dt.String]

    output_dtype = dt.boolean
    output_shape = rlz.shape_like('arg')


@public
class StringSQLLike(FuzzySearch):
    arg: Value[dt.String]
    pattern: Value[dt.String]
    escape: Optional[str] = None


@public
class StringSQLILike(StringSQLLike):
    """SQL ilike operation."""


@public
class RegexSearch(FuzzySearch):
    pass


@public
class RegexExtract(Value):
    arg: Value[dt.String]
    pattern: Value[dt.String]
    index: Value[dt.Integer]

    output_shape = rlz.shape_like("arg")
    output_dtype = dt.string


@public
class RegexReplace(Value):
    arg: Value[dt.String]
    pattern: Value[dt.String]
    replacement: Value[dt.String]

    output_shape = rlz.shape_like("arg")
    output_dtype = dt.string


@public
class StringReplace(Value):
    arg: Value[dt.String]
    pattern: Value[dt.String]
    replacement: Value[dt.String]

    output_shape = rlz.shape_like("arg")
    output_dtype = dt.string


@public
class StringSplit(Value):
    arg: Value[dt.String]
    delimiter: Value[dt.String]

    output_shape = rlz.shape_like("arg")
    output_dtype = dt.Array(dt.string)


@public
class StringConcat(Value):
    arg: VarTuple[Value[dt.String]]

    output_shape = rlz.shape_like('arg')
    output_dtype = rlz.dtype_like('arg')


@public
class ExtractURLField(Value):
    arg: Value[dt.String]

    output_shape = rlz.shape_like("arg")
    output_dtype = dt.string


@public
class ExtractProtocol(ExtractURLField):
    pass


@public
class ExtractAuthority(ExtractURLField):
    pass


@public
class ExtractUserInfo(ExtractURLField):
    pass


@public
class ExtractHost(ExtractURLField):
    pass


@public
class ExtractFile(ExtractURLField):
    pass


@public
class ExtractPath(ExtractURLField):
    pass


@public
class ExtractQuery(ExtractURLField):
    key: Optional[Value[dt.String]] = None


@public
class ExtractFragment(ExtractURLField):
    pass


@public
class StringLength(StringUnary):
    output_dtype = dt.int32


@public
class StringAscii(StringUnary):
    output_dtype = dt.int32


@public
class StringContains(Value):
    haystack: Value[dt.String]
    needle: Value[dt.String]

    output_shape = rlz.shape_like("args")
    output_dtype = dt.bool
