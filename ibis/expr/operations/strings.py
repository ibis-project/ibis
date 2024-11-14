"""String operations."""

from __future__ import annotations

from typing import Optional

from public import public

import ibis.expr.datatypes as dt
import ibis.expr.rules as rlz
from ibis.common.annotations import attribute
from ibis.common.typing import VarTuple  # noqa: TC001
from ibis.expr.operations.core import Unary, Value


@public
class StringUnary(Unary):
    """Base class for string operations accepting one argument."""

    arg: Value[dt.String]

    dtype = dt.string


@public
class Uppercase(StringUnary):
    """Convert a string to uppercase."""


@public
class Lowercase(StringUnary):
    """Convert a string to lowercase."""


@public
class Reverse(StringUnary):
    """Reverse a string."""


@public
class Strip(StringUnary):
    """Strip leading and trailing whitespace."""


@public
class LStrip(StringUnary):
    """Strip leading whitespace."""


@public
class RStrip(StringUnary):
    """Strip trailing whitespace."""


@public
class Capitalize(StringUnary):
    """Capitalize the first letter of a string."""


@public
class Substring(Value):
    """Extract a substring from a string."""

    arg: Value[dt.String]
    start: Value[dt.Integer]
    length: Optional[Value[dt.Integer]] = None

    dtype = dt.string
    shape = rlz.shape_like("args")


@public
class StringSlice(Value):
    """Extract a substring from a string."""

    arg: Value[dt.String]
    start: Optional[Value[dt.Integer]] = None
    end: Optional[Value[dt.Integer]] = None

    dtype = dt.string
    shape = rlz.shape_like("args")


@public
class StrRight(Value):
    """Extract a substring starting from the right of a string."""

    arg: Value[dt.String]
    nchars: Value[dt.Integer]

    shape = rlz.shape_like("args")
    dtype = dt.string


@public
class Repeat(Value):
    """Repeat a string."""

    arg: Value[dt.String]
    times: Value[dt.Integer]

    shape = rlz.shape_like("args")
    dtype = dt.string


@public
class StringFind(Value):
    """Find the position of a substring in a string."""

    arg: Value[dt.String]
    substr: Value[dt.String]
    start: Optional[Value[dt.Integer]] = None
    end: Optional[Value[dt.Integer]] = None

    shape = rlz.shape_like("args")
    dtype = dt.int64


@public
class Translate(Value):
    """Translate characters in a string."""

    arg: Value[dt.String]
    from_str: Value[dt.String]
    to_str: Value[dt.String]

    shape = rlz.shape_like("args")
    dtype = dt.string


@public
class LPad(Value):
    """Pad a string on the left."""

    arg: Value[dt.String]
    length: Value[dt.Integer]
    pad: Optional[Value[dt.String]] = None

    shape = rlz.shape_like("args")
    dtype = dt.string


@public
class RPad(Value):
    """Pad a string on the right."""

    arg: Value[dt.String]
    length: Value[dt.Integer]
    pad: Optional[Value[dt.String]] = None

    shape = rlz.shape_like("args")
    dtype = dt.string


@public
class FindInSet(Value):
    """Find the position of a string in a list of comma-separated strings."""

    needle: Value[dt.String]
    values: VarTuple[Value[dt.String]]

    shape = rlz.shape_like("needle")
    dtype = dt.int64


@public
class StringJoin(Value):
    """Join strings with a separator."""

    arg: VarTuple[Value[dt.String]]
    sep: Value[dt.String]

    dtype = dt.string

    @attribute
    def shape(self):
        return rlz.highest_precedence_shape((self.sep, *self.arg))


@public
class ArrayStringJoin(Value):
    """Join strings in an array with a separator."""

    arg: Value[dt.Array[dt.String]]
    sep: Value[dt.String]

    dtype = dt.string
    shape = rlz.shape_like("args")


@public
class StartsWith(Value):
    """Check if a string starts with another string."""

    arg: Value[dt.String]
    start: Value[dt.String]

    dtype = dt.boolean
    shape = rlz.shape_like("args")


@public
class EndsWith(Value):
    """Check if a string ends with another string."""

    arg: Value[dt.String]
    end: Value[dt.String]

    dtype = dt.boolean
    shape = rlz.shape_like("args")


@public
class FuzzySearch(Value):
    arg: Value[dt.String]
    pattern: Value[dt.String]

    dtype = dt.boolean
    shape = rlz.shape_like("args")


@public
class StringSQLLike(FuzzySearch):
    """SQL LIKE string match operation.

    Similar to globbing.
    """

    arg: Value[dt.String]
    pattern: Value[dt.String]
    escape: Optional[str] = None


@public
class StringSQLILike(StringSQLLike):
    """Case-insensitive SQL LIKE string match operation.

    Similar to case-insensitive globbing.
    """


@public
class RegexSearch(FuzzySearch):
    """Search a string with a regular expression."""


@public
class RegexExtract(Value):
    """Extract a substring from a string using a regular expression."""

    arg: Value[dt.String]
    pattern: Value[dt.String]
    index: Value[dt.Integer]

    shape = rlz.shape_like("args")
    dtype = dt.string


@public
class RegexSplit(Value):
    """Split a string using a regular expression."""

    arg: Value[dt.String]
    pattern: Value[dt.String]

    shape = rlz.shape_like("args")
    dtype = dt.Array(dt.string)


@public
class RegexReplace(Value):
    """Replace a substring in a string using a regular expression."""

    arg: Value[dt.String]
    pattern: Value[dt.String]
    replacement: Value[dt.String]

    shape = rlz.shape_like("args")
    dtype = dt.string


@public
class StringReplace(Value):
    """Replace a substring in a string with another string."""

    arg: Value[dt.String]
    pattern: Value[dt.String]
    replacement: Value[dt.String]

    shape = rlz.shape_like("args")
    dtype = dt.string


@public
class StringSplit(Value):
    """Split a string using a delimiter."""

    arg: Value[dt.String]
    delimiter: Value[dt.String]

    shape = rlz.shape_like("args")
    dtype = dt.Array(dt.string)


@public
class StringConcat(Value):
    """Concatenate strings."""

    arg: VarTuple[Value[dt.String]]

    shape = rlz.shape_like("arg")
    dtype = rlz.dtype_like("arg")


@public
class ExtractProtocol(StringUnary):
    """Extract the protocol from a URL."""


@public
class ExtractAuthority(StringUnary):
    """Extract the authority from a URL."""


@public
class ExtractUserInfo(StringUnary):
    """Extract the user info from a URL."""


@public
class ExtractHost(StringUnary):
    """Extract the host from a URL."""


@public
class ExtractFile(StringUnary):
    """Extract the file from a URL."""


@public
class ExtractPath(StringUnary):
    """Extract the path from a URL."""


@public
class ExtractQuery(StringUnary):
    """Extract the query from a URL."""

    key: Optional[Value[dt.String]] = None


@public
class ExtractFragment(StringUnary):
    """Extract the fragment from a URL."""


@public
class StringLength(StringUnary):
    """Compute the length of a string."""

    dtype = dt.int32


@public
class StringAscii(StringUnary):
    """Compute the ASCII code of the first character of a string."""

    dtype = dt.int32


@public
class StringContains(Value):
    """Check if a string contains a substring."""

    haystack: Value[dt.String]
    needle: Value[dt.String]

    shape = rlz.shape_like("args")
    dtype = dt.bool


@public
class Levenshtein(Value):
    """Compute the Levenshtein distance between two strings."""

    left: Value[dt.String]
    right: Value[dt.String]

    dtype = dt.int64
    shape = rlz.shape_like("args")
