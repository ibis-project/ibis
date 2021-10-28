from public import public

from .. import datatypes as dt
from .. import rules as rlz
from ..signature import Argument as Arg
from .core import UnaryOp, ValueOp
from .generic import BooleanValueOp


@public
class StringUnaryOp(UnaryOp):
    arg = Arg(rlz.string)
    output_type = rlz.shape_like('arg', dt.string)


@public
class Uppercase(StringUnaryOp):
    """Convert string to all uppercase"""


@public
class Lowercase(StringUnaryOp):
    """Convert string to all lowercase"""


@public
class Reverse(StringUnaryOp):
    """Reverse string"""


@public
class Strip(StringUnaryOp):
    """Remove whitespace from left and right sides of string"""


@public
class LStrip(StringUnaryOp):
    """Remove whitespace from left side of string"""


@public
class RStrip(StringUnaryOp):
    """Remove whitespace from right side of string"""


@public
class Capitalize(StringUnaryOp):
    """Return a capitalized version of input string"""


@public
class Substring(ValueOp):
    arg = Arg(rlz.string)
    start = Arg(rlz.integer)
    length = Arg(rlz.integer, default=None)
    output_type = rlz.shape_like('arg', dt.string)


@public
class StrRight(ValueOp):
    arg = Arg(rlz.string)
    nchars = Arg(rlz.integer)
    output_type = rlz.shape_like('arg', dt.string)


@public
class Repeat(ValueOp):
    arg = Arg(rlz.string)
    times = Arg(rlz.integer)
    output_type = rlz.shape_like('arg', dt.string)


@public
class StringFind(ValueOp):
    arg = Arg(rlz.string)
    substr = Arg(rlz.string)
    start = Arg(rlz.integer, default=None)
    end = Arg(rlz.integer, default=None)
    output_type = rlz.shape_like('arg', dt.int64)


@public
class Translate(ValueOp):
    arg = Arg(rlz.string)
    from_str = Arg(rlz.string)
    to_str = Arg(rlz.string)
    output_type = rlz.shape_like('arg', dt.string)


@public
class LPad(ValueOp):
    arg = Arg(rlz.string)
    length = Arg(rlz.integer)
    pad = Arg(rlz.string, default=None)
    output_type = rlz.shape_like('arg', dt.string)


@public
class RPad(ValueOp):
    arg = Arg(rlz.string)
    length = Arg(rlz.integer)
    pad = Arg(rlz.string, default=None)
    output_type = rlz.shape_like('arg', dt.string)


@public
class FindInSet(ValueOp):
    needle = Arg(rlz.string)
    values = Arg(rlz.value_list_of(rlz.string, min_length=1))
    output_type = rlz.shape_like('needle', dt.int64)


@public
class StringJoin(ValueOp):
    sep = Arg(rlz.string)
    arg = Arg(rlz.value_list_of(rlz.string, min_length=1))

    def output_type(self):
        return rlz.shape_like(tuple(self.flat_args()), dt.string)


@public
class StartsWith(ValueOp):
    arg = Arg(rlz.string)
    start = Arg(rlz.string)
    output_type = rlz.shape_like("arg", dt.boolean)


@public
class EndsWith(ValueOp):
    arg = Arg(rlz.string)
    end = Arg(rlz.string)
    output_type = rlz.shape_like("arg", dt.boolean)


@public
class FuzzySearch(ValueOp, BooleanValueOp):
    arg = Arg(rlz.string)
    pattern = Arg(rlz.string)
    output_type = rlz.shape_like('arg', dt.boolean)


@public
class StringSQLLike(FuzzySearch):
    arg = Arg(rlz.string)
    pattern = Arg(rlz.string)
    escape = Arg(str, default=None)


@public
class StringSQLILike(StringSQLLike):
    """SQL ilike operation"""


@public
class RegexSearch(FuzzySearch):
    pass


@public
class RegexExtract(ValueOp):
    arg = Arg(rlz.string)
    pattern = Arg(rlz.string)
    index = Arg(rlz.integer)
    output_type = rlz.shape_like('arg', dt.string)


@public
class RegexReplace(ValueOp):
    arg = Arg(rlz.string)
    pattern = Arg(rlz.string)
    replacement = Arg(rlz.string)
    output_type = rlz.shape_like('arg', dt.string)


@public
class StringReplace(ValueOp):
    arg = Arg(rlz.string)
    pattern = Arg(rlz.string)
    replacement = Arg(rlz.string)
    output_type = rlz.shape_like('arg', dt.string)


@public
class StringSplit(ValueOp):
    arg = Arg(rlz.string)
    delimiter = Arg(rlz.string)
    output_type = rlz.shape_like('arg', dt.Array(dt.string))


@public
class StringConcat(ValueOp):
    arg = Arg(rlz.value_list_of(rlz.string))
    output_type = rlz.shape_like('arg', dt.string)


@public
class ParseURL(ValueOp):
    arg = Arg(rlz.string)
    extract = Arg(
        rlz.isin(
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
    )
    key = Arg(rlz.string, default=None)
    output_type = rlz.shape_like('arg', dt.string)


@public
class StringLength(UnaryOp):
    """Compute the length of a string."""

    output_type = rlz.shape_like('arg', dt.int32)


@public
class StringAscii(UnaryOp):
    output_type = rlz.shape_like('arg', dt.int32)
