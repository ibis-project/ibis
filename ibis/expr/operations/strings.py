from public import public

from ibis.expr import datatypes as dt
from ibis.expr import rules as rlz
from ibis.expr.operations.core import Unary, Value


@public
class StringUnary(Unary):
    arg = rlz.string
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
    arg = rlz.string
    start = rlz.integer
    length = rlz.optional(rlz.integer)

    output_dtype = dt.string
    output_shape = rlz.shape_like('arg')


@public
class StrRight(Value):
    arg = rlz.string
    nchars = rlz.integer
    output_shape = rlz.shape_like("arg")
    output_dtype = dt.string


@public
class Repeat(Value):
    arg = rlz.string
    times = rlz.integer
    output_shape = rlz.shape_like("arg")
    output_dtype = dt.string


@public
class StringFind(Value):
    arg = rlz.string
    substr = rlz.string
    start = rlz.optional(rlz.integer)
    end = rlz.optional(rlz.integer)

    output_shape = rlz.shape_like("arg")
    output_dtype = dt.int64


@public
class Translate(Value):
    arg = rlz.string
    from_str = rlz.string
    to_str = rlz.string

    output_shape = rlz.shape_like("arg")
    output_dtype = dt.string


@public
class LPad(Value):
    arg = rlz.string
    length = rlz.integer
    pad = rlz.optional(rlz.string)

    output_shape = rlz.shape_like("arg")
    output_dtype = dt.string


@public
class RPad(Value):
    arg = rlz.string
    length = rlz.integer
    pad = rlz.optional(rlz.string)

    output_shape = rlz.shape_like("arg")
    output_dtype = dt.string


@public
class FindInSet(Value):
    needle = rlz.string
    values = rlz.value_list_of(rlz.string, min_length=1)

    output_shape = rlz.shape_like("needle")
    output_dtype = dt.int64


@public
class StringJoin(Value):
    sep = rlz.string
    arg = rlz.value_list_of(rlz.string, min_length=1)

    output_dtype = dt.string
    output_shape = rlz.shape_like("arg")


@public
class StartsWith(Value):
    arg = rlz.string
    start = rlz.scalar(rlz.string)
    output_dtype = dt.boolean
    output_shape = rlz.shape_like("arg")


@public
class EndsWith(Value):
    arg = rlz.string
    end = rlz.scalar(rlz.string)
    output_dtype = dt.boolean
    output_shape = rlz.shape_like("arg")


@public
class FuzzySearch(Value):
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
class RegexExtract(Value):
    arg = rlz.string
    pattern = rlz.string
    index = rlz.integer

    output_shape = rlz.shape_like("arg")
    output_dtype = dt.string


@public
class RegexReplace(Value):
    arg = rlz.string
    pattern = rlz.string
    replacement = rlz.string

    output_shape = rlz.shape_like("arg")
    output_dtype = dt.string


@public
class StringReplace(Value):
    arg = rlz.string
    pattern = rlz.string
    replacement = rlz.string

    output_shape = rlz.shape_like("arg")
    output_dtype = dt.string


@public
class StringSplit(Value):
    arg = rlz.string
    delimiter = rlz.string

    output_shape = rlz.shape_like("arg")
    output_dtype = dt.Array(dt.string)


@public
class StringConcat(Value):
    arg = rlz.value_list_of(rlz.string)

    output_shape = rlz.shape_like("arg")
    output_dtype = dt.string


@public
class ParseURL(Value):
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
class StringLength(Unary):
    output_dtype = dt.int32


@public
class StringAscii(Unary):
    output_dtype = dt.int32


@public
class StringContains(Value):
    haystack = rlz.string
    needle = rlz.string

    output_shape = rlz.shape_like("args")
    output_dtype = dt.bool
