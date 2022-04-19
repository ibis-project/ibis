from public import public

from ibis.expr import datatypes as dt
from ibis.expr import rules as rlz
from ibis.expr.operations.core import UnaryOp, ValueOp


@public
class StringUnaryOp(UnaryOp):
    arg = rlz.string
    output_dtype = dt.string


@public
class Uppercase(StringUnaryOp):
    pass


@public
class Lowercase(StringUnaryOp):
    pass


@public
class Reverse(StringUnaryOp):
    pass


@public
class Strip(StringUnaryOp):
    pass


@public
class LStrip(StringUnaryOp):
    pass


@public
class RStrip(StringUnaryOp):
    pass


@public
class Capitalize(StringUnaryOp):
    pass


@public
class Substring(ValueOp):
    arg = rlz.string
    start = rlz.integer
    length = rlz.optional(rlz.integer)

    output_dtype = dt.string
    output_shape = rlz.shape_like('arg')


@public
class StrRight(ValueOp):
    arg = rlz.string
    nchars = rlz.integer
    output_shape = rlz.shape_like("arg")
    output_dtype = dt.string


@public
class Repeat(ValueOp):
    arg = rlz.string
    times = rlz.integer
    output_shape = rlz.shape_like("arg")
    output_dtype = dt.string


@public
class StringFind(ValueOp):
    arg = rlz.string
    substr = rlz.string
    start = rlz.optional(rlz.integer)
    end = rlz.optional(rlz.integer)

    output_shape = rlz.shape_like("arg")
    output_dtype = dt.int64


@public
class Translate(ValueOp):
    arg = rlz.string
    from_str = rlz.string
    to_str = rlz.string

    output_shape = rlz.shape_like("arg")
    output_dtype = dt.string


@public
class LPad(ValueOp):
    arg = rlz.string
    length = rlz.integer
    pad = rlz.optional(rlz.string)

    output_shape = rlz.shape_like("arg")
    output_dtype = dt.string


@public
class RPad(ValueOp):
    arg = rlz.string
    length = rlz.integer
    pad = rlz.optional(rlz.string)

    output_shape = rlz.shape_like("arg")
    output_dtype = dt.string


@public
class FindInSet(ValueOp):
    needle = rlz.string
    values = rlz.value_list_of(rlz.string, min_length=1)

    output_shape = rlz.shape_like("needle")
    output_dtype = dt.int64


@public
class StringJoin(ValueOp):
    sep = rlz.string
    arg = rlz.value_list_of(rlz.string, min_length=1)

    output_dtype = dt.string
    output_shape = rlz.shape_like("arg")


@public
class StartsWith(ValueOp):
    arg = rlz.string
    start = rlz.scalar(rlz.string)
    output_dtype = dt.boolean
    output_shape = rlz.shape_like("arg")


@public
class EndsWith(ValueOp):
    arg = rlz.string
    end = rlz.scalar(rlz.string)
    output_dtype = dt.boolean
    output_shape = rlz.shape_like("arg")


@public
class FuzzySearch(ValueOp):
    arg = rlz.string
    pattern = rlz.string
    output_dtype = dt.boolean
    output_shape = rlz.shape_like('arg')


@public
class StringSQLLike(FuzzySearch):
    arg = rlz.string
    pattern = rlz.string
    escape = rlz.optional(rlz.instance_of(str))


@public
class StringSQLILike(StringSQLLike):
    """SQL ilike operation"""


@public
class RegexSearch(FuzzySearch):
    pass


@public
class RegexExtract(ValueOp):
    arg = rlz.string
    pattern = rlz.string
    index = rlz.integer

    output_shape = rlz.shape_like("arg")
    output_dtype = dt.string


@public
class RegexReplace(ValueOp):
    arg = rlz.string
    pattern = rlz.string
    replacement = rlz.string

    output_shape = rlz.shape_like("arg")
    output_dtype = dt.string


@public
class StringReplace(ValueOp):
    arg = rlz.string
    pattern = rlz.string
    replacement = rlz.string

    output_shape = rlz.shape_like("arg")
    output_dtype = dt.string


@public
class StringSplit(ValueOp):
    arg = rlz.string
    delimiter = rlz.string

    output_shape = rlz.shape_like("arg")
    output_dtype = dt.Array(dt.string)


@public
class StringConcat(ValueOp):
    arg = rlz.value_list_of(rlz.string)

    output_shape = rlz.shape_like("arg")
    output_dtype = dt.string


@public
class ParseURL(ValueOp):
    arg = rlz.string
    extract = rlz.isin(
        {
            'PROTOCOL',
            'HOST',
            'PATH',
            'REF',
            'AUTHORITY',
            'FILE',
            'USERINFO',
            'QUERY',
        }
    )
    key = rlz.optional(rlz.string)

    output_shape = rlz.shape_like("arg")
    output_dtype = dt.string


@public
class StringLength(UnaryOp):
    output_dtype = dt.int32


@public
class StringAscii(UnaryOp):
    output_dtype = dt.int32


@public
class StringContains(ValueOp):
    haystack = rlz.string
    needle = rlz.string

    output_shape = rlz.shape_like("args")
    output_dtype = dt.bool
