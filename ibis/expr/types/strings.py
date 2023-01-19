from __future__ import annotations

import functools
import operator
from typing import TYPE_CHECKING, Any, Iterable, Literal, Sequence

from public import public

import ibis.expr.operations as ops
from ibis import util
from ibis.expr.types.core import _binop
from ibis.expr.types.generic import Column, Scalar, Value

if TYPE_CHECKING:
    from ibis.expr import types as ir


@public
class StringValue(Value):
    def __getitem__(self, key: slice | int | ir.IntegerValue) -> StringValue:
        from ibis.expr import types as ir

        if isinstance(key, slice):
            start, stop, step = key.start, key.stop, key.step

            if step is not None and not isinstance(step, ir.Expr) and step != 1:
                raise ValueError('Step can only be 1')

            if not isinstance(start, ir.Expr):
                if start is not None and start < 0:
                    raise ValueError(
                        "Negative slicing not yet supported, got start value "
                        f"of {start:d}"
                    )
                if start is None:
                    start = 0

            if not isinstance(stop, ir.Expr):
                if stop is not None and stop < 0:
                    raise ValueError(
                        "Negative slicing not yet supported, got stop value "
                        f"of {stop:d}"
                    )
                if stop is None:
                    stop = self.length()

            return self.substr(start, stop - start)
        elif isinstance(key, int):
            return self.substr(key, 1)
        raise NotImplementedError(f"string __getitem__[{key.__class__.__name__}]")

    def length(self) -> ir.IntegerValue:
        """Compute the length of a string.

        Returns
        -------
        IntegerValue
            The length of the input
        """
        return ops.StringLength(self).to_expr()

    def lower(self) -> StringValue:
        """Convert string to all lowercase.

        Returns
        -------
        StringValue
            Lowercase string
        """
        return ops.Lowercase(self).to_expr()

    def upper(self) -> StringValue:
        """Convert string to all uppercase.

        Returns
        -------
        StringValue
            Uppercase string
        """
        return ops.Uppercase(self).to_expr()

    def reverse(self) -> StringValue:
        """Reverse the characters of a string.

        Returns
        -------
        StringValue
            Reversed string
        """
        return ops.Reverse(self).to_expr()

    def ascii_str(self) -> ir.IntegerValue:
        """Return the numeric ASCII code of the first character of a string.

        Returns
        -------
        IntegerValue
            ASCII code of the first character of the input
        """
        return ops.StringAscii(self).to_expr()

    def strip(self) -> StringValue:
        """Remove whitespace from left and right sides of a string.

        Returns
        -------
        StringValue
            Stripped string
        """
        return ops.Strip(self).to_expr()

    def lstrip(self) -> StringValue:
        """Remove whitespace from the left side of string.

        Returns
        -------
        StringValue
            Left-stripped string
        """
        return ops.LStrip(self).to_expr()

    def rstrip(self) -> StringValue:
        """Remove whitespace from the right side of string.

        Returns
        -------
        StringValue
            Right-stripped string
        """
        return ops.RStrip(self).to_expr()

    def capitalize(self) -> StringValue:
        """Capitalize the input string.

        Returns
        -------
        StringValue
            Capitalized string
        """
        return ops.Capitalize(self).to_expr()

    initcap = capitalize

    def __contains__(self, *_: Any) -> bool:
        raise TypeError("Use string_expr.contains(arg)")

    def contains(self, substr: str | StringValue) -> ir.BooleanValue:
        """Return whether the expression contains `substr`.

        Parameters
        ----------
        substr
            Substring for which to check

        Returns
        -------
        BooleanValue
            Boolean indicating the presence of `substr` in the expression
        """
        return ops.StringContains(self, substr).to_expr()

    def hashbytes(
        self,
        how: Literal["md5", "sha1", "sha256", "sha512"] = "sha256",
    ) -> ir.BinaryValue:
        """Compute the binary hash value of the input.

        Parameters
        ----------
        how
            Hash algorithm to use

        Returns
        -------
        BinaryValue
            Binary expression
        """
        return ops.HashBytes(self, how).to_expr()

    def substr(
        self,
        start: int | ir.IntegerValue,
        length: int | ir.IntegerValue | None = None,
    ) -> StringValue:
        """Extract a substring.

        Parameters
        ----------
        start
            First character to start splitting, indices start at 0
        length
            Maximum length of each substring. If not supplied, searches the
            entire string

        Returns
        -------
        StringValue
            Found substring
        """
        return ops.Substring(self, start, length).to_expr()

    def left(self, nchars: int | ir.IntegerValue) -> StringValue:
        """Return the `nchars` left-most characters.

        Parameters
        ----------
        nchars
            Maximum number of characters to return

        Returns
        -------
        StringValue
            Characters from the start
        """
        return self.substr(0, length=nchars)

    def right(self, nchars: int | ir.IntegerValue) -> StringValue:
        """Return up to `nchars` from the end of each string.

        Parameters
        ----------
        nchars
            Maximum number of characters to return

        Returns
        -------
        StringValue
            Characters from the end
        """
        return ops.StrRight(self, nchars).to_expr()

    def repeat(self, n: int | ir.IntegerValue) -> StringValue:
        """Repeat a string `n` times.

        Parameters
        ----------
        n
            Number of repetitions

        Returns
        -------
        StringValue
            Repeated string
        """
        return ops.Repeat(self, n).to_expr()

    __mul__ = __rmul__ = repeat

    def translate(self, from_str: StringValue, to_str: StringValue) -> StringValue:
        """Replace `from_str` characters in `self` characters in `to_str`.

        To avoid unexpected behavior, `from_str` should be shorter than
        `to_str`.

        Parameters
        ----------
        from_str
            Characters in `arg` to replace
        to_str
            Characters to use for replacement

        Returns
        -------
        StringValue
            Translated string

        Examples
        --------
        >>> import ibis
        >>> table = ibis.table(dict(string_col='string'))
        >>> result = table.string_col.translate('a', 'b')
        """
        return ops.Translate(self, from_str, to_str).to_expr()

    def find(
        self,
        substr: str | StringValue,
        start: int | ir.IntegerValue | None = None,
        end: int | ir.IntegerValue | None = None,
    ) -> ir.IntegerValue:
        """Return the position of the first occurence of substring.

        Parameters
        ----------
        substr
            Substring to search for
        start
            Zero based index of where to start the search
        end
            Zero based index of where to stop the search. Currently not
            implemented.

        Returns
        -------
        IntegerValue
            Position of `substr` in `arg` starting from `start`
        """
        if end is not None:
            raise NotImplementedError
        return ops.StringFind(self, substr, start, end).to_expr()

    def lpad(
        self,
        length: int | ir.IntegerValue,
        pad: str | StringValue = " ",
    ) -> StringValue:
        """Pad `arg` by truncating on the right or padding on the left.

        Parameters
        ----------
        length
            Length of output string
        pad
            Pad character

        Returns
        -------
        StringValue
            Padded string

        Examples
        --------
        >>> import ibis
        >>> short_str = ibis.literal("a")
        >>> result = short_str.lpad(5, "-")  # ----a
        >>> long_str = ibis.literal("abcdefg")
        >>> result = long_str.lpad(5, "-")  # abcde
        """
        return ops.LPad(self, length, pad).to_expr()

    def rpad(
        self,
        length: int | ir.IntegerValue,
        pad: str | StringValue = " ",
    ) -> StringValue:
        """Pad `self` by truncating or padding on the right.

        Parameters
        ----------
        self
            String to pad
        length
            Length of output string
        pad
            Pad character

        Examples
        --------
        >>> import ibis
        >>> short_str = ibis.literal("a")
        >>> result = short_str.lpad(5, "-")  # a----
        >>> long_str = ibis.literal("abcdefg")
        >>> result = long_str.lpad(5, "-")  # abcde

        Returns
        -------
        StringValue
            Padded string
        """
        return ops.RPad(self, length, pad).to_expr()

    def find_in_set(self, str_list: Sequence[str]) -> ir.IntegerValue:
        """Find the first occurence of `str_list` within a list of strings.

        No string in `str_list` can have a comma.

        Parameters
        ----------
        str_list
            Sequence of strings

        Examples
        --------
        >>> import ibis
        >>> table = ibis.table(dict(string_col='string'))
        >>> result = table.string_col.find_in_set(['a', 'b'])

        Returns
        -------
        IntegerValue
            Position of `str_list` in `self`. Returns -1 if `self` isn't found
            or if `self` contains `','`.
        """
        return ops.FindInSet(self, str_list).to_expr()

    def join(self, strings: Sequence[str | StringValue]) -> StringValue:
        """Join a list of strings using `self` as the separator.

        Parameters
        ----------
        strings
            Strings to join with `arg`

        Examples
        --------
        >>> import ibis
        >>> sep = ibis.literal(',')
        >>> result = sep.join(['a', 'b', 'c'])

        Returns
        -------
        StringValue
            Joined string
        """
        return ops.StringJoin(self, strings).to_expr()

    def startswith(self, start: str | StringValue) -> ir.BooleanValue:
        """Determine whether `self` starts with `end`.

        Parameters
        ----------
        start
            prefix to check for

        Examples
        --------
        >>> import ibis
        >>> text = ibis.literal('Ibis project')
        >>> result = text.startswith('Ibis')

        Returns
        -------
        BooleanValue
            Boolean indicating whether `self` starts with `start`
        """
        return ops.StartsWith(self, start).to_expr()

    def endswith(self, end: str | StringValue) -> ir.BooleanValue:
        """Determine if `self` ends with `end`.

        Parameters
        ----------
        end
            Suffix to check for

        Examples
        --------
        >>> import ibis
        >>> text = ibis.literal('Ibis project')
        >>> result = text.endswith('project')

        Returns
        -------
        BooleanValue
            Boolean indicating whether `self` ends with `end`
        """
        return ops.EndsWith(self, end).to_expr()

    def like(
        self,
        patterns: str | StringValue | Iterable[str | StringValue],
    ) -> ir.BooleanValue:
        """Match `patterns` against `self`, case-sensitive.

        This function is modeled after the SQL `LIKE` directive. Use `%` as a
        multiple-character wildcard or `_` as a single-character wildcard.

        Use `re_search` or `rlike` for regular expression-based matching.

        Parameters
        ----------
        patterns
            If `pattern` is a list, then if any pattern matches the input then
            the corresponding row in the output is `True`.

        Returns
        -------
        BooleanValue
            Column indicating matches
        """
        return functools.reduce(
            operator.or_,
            (
                ops.StringSQLLike(self, pattern).to_expr()
                for pattern in util.promote_list(patterns)
            ),
        )

    def ilike(
        self,
        patterns: str | StringValue | Iterable[str | StringValue],
    ) -> ir.BooleanValue:
        """Match `patterns` against `self`, case-insensitive.

        This function is modeled after SQL's `ILIKE` directive. Use `%` as a
        multiple-character wildcard or `_` as a single-character wildcard.

        Use `re_search` or `rlike` for regular expression-based matching.

        Parameters
        ----------
        patterns
            If `pattern` is a list, then if any pattern matches the input then
            the corresponding row in the output is `True`.

        Returns
        -------
        BooleanValue
            Column indicating matches
        """
        return functools.reduce(
            operator.or_,
            (
                ops.StringSQLILike(self, pattern).to_expr()
                for pattern in util.promote_list(patterns)
            ),
        )

    @util.backend_sensitive(
        why="Different backends support different regular expression syntax."
    )
    def re_search(self, pattern: str | StringValue) -> ir.BooleanValue:
        """Return whether the values match `pattern`.

        Returns `True` if the regex matches a string and `False` otherwise.

        Parameters
        ----------
        pattern
            Regular expression use for searching

        Returns
        -------
        BooleanValue
            Indicator of matches
        """
        return ops.RegexSearch(self, pattern).to_expr()

    rlike = re_search

    @util.backend_sensitive(
        why="Different backends support different regular expression syntax."
    )
    def re_extract(
        self,
        pattern: str | StringValue,
        index: int | ir.IntegerValue,
    ) -> StringValue:
        """Return the specified match at `index` from a regex `pattern`.

        Parameters
        ----------
        pattern
            Reguar expression pattern string
        index
            The index of the match group to return.

            The behavior of this function follows the behavior of Python's
            [`re.match`](https://docs.python.org/3/library/re.html#match-objects):
            when `index` is zero and there's a match, return the entire string,
            otherwise return the content of the `index`-th match group.

        Returns
        -------
        StringValue
            Extracted match or whole string if `index` is zero
        """
        return ops.RegexExtract(self, pattern, index).to_expr()

    @util.backend_sensitive(
        why="Different backends support different regular expression syntax."
    )
    def re_replace(
        self,
        pattern: str | StringValue,
        replacement: str | StringValue,
    ) -> StringValue:
        r"""Replace match found by regex `pattern` with `replacement`.

        Parameters
        ----------
        pattern
            Regular expression string
        replacement
            Replacement string or regular expression

        Examples
        --------
        >>> import ibis
        >>> str_literal = ibis.literal("aaabbbaaa")
        >>> result = str_literal.re_replace("(b+)", r"<\1>")  # aaa<bbb>aaa

        Returns
        -------
        StringValue
            Modified string
        """
        return ops.RegexReplace(self, pattern, replacement).to_expr()

    def replace(
        self,
        pattern: StringValue,
        replacement: StringValue,
    ) -> StringValue:
        """Replace each exact match of `pattern` with `replacement`.

        Parameters
        ----------
        pattern
            String pattern
        replacement
            String replacement

        Examples
        --------
        >>> import ibis
        >>> str_literal = ibis.literal("aaabbbaaa")
        >>> result = str_literal.replace("aaa", "ccc")  # cccbbbccc

        Returns
        -------
        StringValue
            Replaced string
        """
        return ops.StringReplace(self, pattern, replacement).to_expr()

    def to_timestamp(self, format_str: str) -> ir.TimestampValue:
        """Parse a string and return a timestamp.

        Parameters
        ----------
        format_str
            Format string in `strptime` format

        Examples
        --------
        >>> import ibis
        >>> date_as_str = ibis.literal('20170206')
        >>> result = date_as_str.to_timestamp('%Y%m%d')

        Returns
        -------
        TimestampValue
            Parsed timestamp value
        """
        return ops.StringToTimestamp(self, format_str).to_expr()

    @util.deprecated(
        as_of='4.0',
        removed_in='5.0',
        instead=(
            'use .protocol(), .authroity(), .userinfo(), .host(), .file(), .path(), .query(), or .fragment().'
        ),
    )
    def parse_url(
        self,
        extract: Literal[
            "PROTOCOL",
            "AUTHORITY",
            "USERINFO",
            "HOST",
            "FILE",
            "PATH",
            "QUERY",
            "REF",
        ],
        key: str | None = None,
    ) -> StringValue:
        """Parse a URL and extract its components.

        `key` can be used to extract query values when `extract == 'QUERY'`

        Parameters
        ----------
        extract
            Component of URL to extract
        key
            Query component to extract

        Examples
        --------
        >>> import ibis
        >>> url = ibis.literal("https://www.youtube.com/watch?v=kEuEcWfewf8&t=10")
        >>> result = url.parse_url('QUERY', 'v')  # kEuEcWfewf

        Returns
        -------
        StringValue
            Extracted string value
        """
        return ops.ParseURL(self, extract, key).to_expr()

    def protocol(self):
        """Parse a URL and extract protocol.

        Examples
        --------
        >>> import ibis
        >>> url = ibis.literal("https://user:pass@example.com:80/docs/books")
        >>> result = url.protocol()  # https

        Returns
        -------
        StringValue
            Extracted string value
        """
        return ops.ExtractProtocol(self).to_expr()

    def authority(self):
        """Parse a URL and extract authority.

        Examples
        --------
        >>> import ibis
        >>> url = ibis.literal("https://user:pass@example.com:80/docs/books")
        >>> result = url.authority()  # user:pass@example.com:80

        Returns
        -------
        StringValue
            Extracted string value
        """
        return ops.ExtractAuthority(self).to_expr()

    def userinfo(self):
        """Parse a URL and extract user info.

        Examples
        --------
        >>> import ibis
        >>> url = ibis.literal("https://user:pass@example.com:80/docs/books")
        >>> result = url.authority()  # user:pass

        Returns
        -------
        StringValue
            Extracted string value
        """
        return ops.ExtractUserInfo(self).to_expr()

    def host(self):
        """Parse a URL and extract host.

        Examples
        --------
        >>> import ibis
        >>> url = ibis.literal("https://user:pass@example.com:80/docs/books")
        >>> result = url.authority()  # example.com

        Returns
        -------
        StringValue
            Extracted string value
        """
        return ops.ExtractHost(self).to_expr()

    def file(self):
        """Parse a URL and extract file.

        Examples
        --------
        >>> import ibis
        >>> url = ibis.literal("https://example.com:80/docs/books/tutorial/index.html?name=networking")
        >>> result = url.authority()  # docs/books/tutorial/index.html?name=networking

        Returns
        -------
        StringValue
            Extracted string value
        """
        return ops.ExtractFile(self).to_expr()

    def path(self):
        """Parse a URL and extract path.

        Examples
        --------
        >>> import ibis
        >>> url = ibis.literal("https://example.com:80/docs/books/tutorial/index.html?name=networking")
        >>> result = url.authority()  # docs/books/tutorial/index.html

        Returns
        -------
        StringValue
            Extracted string value
        """
        return ops.ExtractPath(self).to_expr()

    def query(self, key: str | StringValue | None = None):
        """Parse a URL and returns query strring or query string parameter.

        If key is passed, return the value of the query string parameter named.
        If key is absent, return the query string.

        Parameters
        ----------
        key
            Query component to extract

        Examples
        --------
        >>> import ibis
        >>> url = ibis.literal("https://example.com:80/docs/books/tutorial/index.html?name=networking")
        >>> result = url.query()  # name=networking
        >>> query_name = url.query('name')  # networking

        Returns
        -------
        StringValue
            Extracted string value
        """
        return ops.ExtractQuery(self, key).to_expr()

    def fragment(self):
        """Parse a URL and extract fragment identifier.

        Examples
        --------
        >>> import ibis
        >>> url = ibis.literal("https://example.com:80/docs/#DOWNLOADING")
        >>> result = url.fragment()  # DOWNLOADING

        Returns
        -------
        StringValue
            Extracted string value
        """
        return ops.ExtractFragment(self).to_expr()

    def split(self, delimiter: str | StringValue) -> ir.ArrayValue:
        """Split as string on `delimiter`.

        Parameters
        ----------
        delimiter
            Value to split by

        Returns
        -------
        ArrayValue
            The string split by `delimiter`
        """
        return ops.StringSplit(self, delimiter).to_expr()

    def concat(
        self,
        other: str | StringValue,
        *args: str | StringValue,
    ) -> StringValue:
        """Concatenate strings.

        Parameters
        ----------
        other
            String to concatenate
        args
            Additional strings to concatenate

        Returns
        -------
        StringValue
            All strings concatenated
        """
        return ops.StringConcat((self, other, *args)).to_expr()

    def __add__(self, other: str | StringValue) -> StringValue:
        """Concatenate strings.

        Parameters
        ----------
        other
            String to concatenate

        Returns
        -------
        StringValue
            All strings concatenated
        """
        return ops.StringConcat((self, other)).to_expr()

    def __radd__(self, other: str | StringValue) -> StringValue:
        """Concatenate strings.

        Parameters
        ----------
        other
            String to concatenate

        Returns
        -------
        StringValue
            All strings concatenated
        """
        return ops.StringConcat((other, self)).to_expr()

    def convert_base(
        self,
        from_base: int | ir.IntegerValue,
        to_base: int | ir.IntegerValue,
    ) -> ir.IntegerValue:
        """Convert a string representing an integer from one base to another.

        Parameters
        ----------
        from_base
            Numeric base of the expression
        to_base
            New base

        Returns
        -------
        IntegerValue
            Converted expression
        """
        return ops.BaseConvert(self, from_base, to_base).to_expr()

    def __mul__(self, n: int | ir.IntegerValue) -> StringValue | NotImplemented:
        return _binop(ops.Repeat, self, n)

    __rmul__ = __mul__


@public
class StringScalar(Scalar, StringValue):
    pass


@public
class StringColumn(Column, StringValue):
    pass
