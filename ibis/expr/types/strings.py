from __future__ import annotations

import functools
import operator
from typing import TYPE_CHECKING, Any, Iterable, Literal, Sequence

if TYPE_CHECKING:
    from ibis.expr import types as ir

from public import public

from ibis import util
from ibis.expr.types.core import _binop
from ibis.expr.types.generic import Column, Scalar, Value


@public
class StringValue(Value):
    def __getitem__(self, key: slice | int | ir.IntegerValue) -> StringValue:
        from ibis.expr import types as ir

        if isinstance(key, slice):
            start, stop, step = key.start, key.stop, key.step

            if (
                step is not None
                and not isinstance(step, ir.Expr)
                and step != 1
            ):
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
        raise NotImplementedError(
            f"string __getitem__[{key.__class__.__name__}]"
        )

    def length(self) -> ir.IntegerValue:
        """Compute the length of a string.

        Returns
        -------
        IntegerValue
            The length of the input
        """
        import ibis.expr.operations as ops

        return ops.StringLength(self).to_expr()

    def lower(self) -> StringValue:
        """Convert string to all lowercase.

        Returns
        -------
        StringValue
            Lowercase string
        """
        import ibis.expr.operations as ops

        return ops.Lowercase(self).to_expr()

    def upper(self) -> StringValue:
        """Convert string to all uppercase.

        Returns
        -------
        StringValue
            Uppercase string
        """
        import ibis.expr.operations as ops

        return ops.Uppercase(self).to_expr()

    def reverse(self) -> StringValue:
        """Reverse the characters of a string.

        Returns
        -------
        StringValue
            Reversed string
        """
        import ibis.expr.operations as ops

        return ops.Reverse(self).to_expr()

    def ascii_str(self) -> ir.IntegerValue:
        """Return the numeric ASCII code of the first character of a string.

        Returns
        -------
        IntegerValue
            ASCII code of the first character of the input
        """
        import ibis.expr.operations as ops

        return ops.StringAscii(self).to_expr()

    def strip(self) -> StringValue:
        """Remove whitespace from left and right sides of a string.

        Returns
        -------
        StringValue
            Stripped string
        """
        import ibis.expr.operations as ops

        return ops.Strip(self).to_expr()

    def lstrip(self) -> StringValue:
        """Remove whitespace from the left side of string.

        Returns
        -------
        StringValue
            Left-stripped string
        """
        import ibis.expr.operations as ops

        return ops.LStrip(self).to_expr()

    def rstrip(self) -> StringValue:
        """Remove whitespace from the right side of string.

        Returns
        -------
        StringValue
            Right-stripped string
        """
        import ibis.expr.operations as ops

        return ops.RStrip(self).to_expr()

    def capitalize(self) -> StringValue:
        """Capitalize the input string.

        Returns
        -------
        StringValue
            Capitalized string
        """
        import ibis.expr.operations as ops

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
        import ibis.expr.operations as ops

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
        import ibis.expr.operations as ops

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
        import ibis.expr.operations as ops

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
        import ibis.expr.operations as ops

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
        import ibis.expr.operations as ops

        return ops.Repeat(self, n).to_expr()

    __mul__ = __rmul__ = repeat

    def translate(
        self, from_str: StringValue, to_str: StringValue
    ) -> StringValue:
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
        >>> expr = table.string_col.translate('a', 'b')
        >>> expr
        r0 := UnboundTable: unbound_table_0
          string_col string
        Translate(r0.string_col, from_str='a', to_str='b')
        >>> expr = table.string_col.translate('a', 'bc')
        >>> expr
        r0 := UnboundTable: unbound_table_0
          string_col string
        Translate(r0.string_col, from_str='a', to_str='bc')
        """
        import ibis.expr.operations as ops

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
        import ibis.expr.operations as ops

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
        >>> table = ibis.table(dict(strings='string'))
        >>> expr = table.strings.lpad(5, '-')
        >>> expr
        r0 := UnboundTable: unbound_table_1
          strings string
        LPad(r0.strings, length=5, pad='-')
        >>> expr = ibis.literal('a').lpad(5, '-')  # 'a' becomes '----a'
        >>> expr
        LPad('a', length=5, pad='-')
        >>> expr = ibis.literal('abcdefg').lpad(5, '-')  # 'abcdefg' becomes 'abcde'
        >>> expr
        LPad('abcdefg', length=5, pad='-')
        """  # noqa: E501
        import ibis.expr.operations as ops

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
        >>> table = ibis.table(dict(string_col='string'))
        >>> expr = table.string_col.rpad(5, '-')
        >>> expr
        r0 := UnboundTable: unbound_table_2
          string_col string
        RPad(r0.string_col, length=5, pad='-')
        >>> expr = ibis.literal('a').rpad(5, '-')  # 'a' becomes 'a----'
        >>> expr
        RPad('a', length=5, pad='-')
        >>> expr = ibis.literal('abcdefg').rpad(5, '-')  # 'abcdefg' becomes 'abcde'
        >>> expr
        RPad('abcdefg', length=5, pad='-')

        Returns
        -------
        StringValue
            Padded string
        """  # noqa: E501
        import ibis.expr.operations as ops

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
        >>> table = ibis.table(dict(strings='string'))
        >>> result = table.strings.find_in_set(['a', 'b'])
        >>> result
        r0 := UnboundTable: unbound_table_0
          strings string
        FindInSet(needle=r0.strings, values=[ValueList(values=['a', 'b'])])

        Returns
        -------
        IntegerValue
            Position of `str_list` in `self`. Returns -1 if `self` isn't found
            or if `self` contains `','`.
        """
        import ibis.expr.operations as ops

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
        >>> result
        StringJoin(sep=',', [ValueList(values=['a', 'b', 'c'])])

        Returns
        -------
        StringValue
            Joined string
        """
        import ibis.expr.operations as ops

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
        >>> text.startswith('Ibis')
        StartsWith('Ibis project', start='Ibis')

        Returns
        -------
        BooleanValue
            Boolean indicating whether `self` starts with `start`
        """
        import ibis.expr.operations as ops

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
        >>> text.endswith('project')
        EndsWith('Ibis project', end='project')

        Returns
        -------
        BooleanValue
            Boolean indicating whether `self` ends with `end`
        """
        import ibis.expr.operations as ops

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
        import ibis.expr.operations as ops

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
        import ibis.expr.operations as ops

        return functools.reduce(
            operator.or_,
            (
                ops.StringSQLILike(self, pattern).to_expr()
                for pattern in util.promote_list(patterns)
            ),
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
        import ibis.expr.operations as ops

        return ops.RegexSearch(self, pattern).to_expr()

    rlike = re_search

    def re_extract(
        self,
        pattern: str | StringValue,
        index: int | ir.IntegerValue,
    ) -> StringValue:
        """Return the specified match at `index` from a regex `pattern`.

        Parameters
        ----------
        pattern
            Reguar expression string
        index
            Zero-based index of match to return

        Returns
        -------
        StringValue
            Extracted match
        """
        import ibis.expr.operations as ops

        return ops.RegexExtract(self, pattern, index).to_expr()

    def re_replace(
        self,
        pattern: str | StringValue,
        replacement: str | StringValue,
    ) -> StringValue:
        """Replace match found by regex `pattern` with `replacement`.

        Parameters
        ----------
        pattern
            Regular expression string
        replacement
            Replacement string or regular expression

        Examples
        --------
        >>> import ibis
        >>> table = ibis.table(dict(strings='string'))
        >>> result = table.strings.replace('(b+)', r'<\1>')  # 'aaabbbaa' becomes 'aaa<bbb>aaa'
        >>> result
        r0 := UnboundTable: unbound_table_1
          strings string
        StringReplace(r0.strings, pattern='(b+)', replacement='<\\1>')

        Returns
        -------
        StringValue
            Modified string
        """  # noqa: E501
        import ibis.expr.operations as ops

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
        >>> table = ibis.table(dict(strings='string'))
        >>> result = table.strings.replace('aaa', 'foo')  # 'aaabbbaaa' becomes 'foobbbfoo'
        >>> result
        r0 := UnboundTable: unbound_table_1
          strings string
        StringReplace(r0.strings, pattern='aaa', replacement='foo')

        Returns
        -------
        StringVulae
            Replaced string
        """  # noqa: E501
        import ibis.expr.operations as ops

        return ops.StringReplace(self, pattern, replacement).to_expr()

    def to_timestamp(
        self, format_str: str, timezone: str | None = None
    ) -> ir.TimestampValue:
        """Parse a string and return a timestamp.

        Parameters
        ----------
        format_str
            Format string in `strptime` format
        timezone
            A string indicating the timezone. For example `'America/New_York'`

        Examples
        --------
        >>> import ibis
        >>> date_as_str = ibis.literal('20170206')
        >>> result = date_as_str.to_timestamp('%Y%m%d')
        >>> result
        StringToTimestamp('20170206', format_str='%Y%m%d')

        Returns
        -------
        TimestampValue
            Parsed timestamp value
        """
        import ibis.expr.operations as ops

        return ops.StringToTimestamp(self, format_str, timezone).to_expr()

    def parse_url(
        self,
        extract: Literal[
            "PROTOCOL",
            "HOST",
            "PATH",
            "REF",
            "AUTHORITY",
            "FILE",
            "USERINFO",
            "QUERY",
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
        >>> url = "https://www.youtube.com/watch?v=kEuEcWfewf8&t=10"
        >>> parse_url(url, 'QUERY', 'v')  # doctest: +SKIP
        'kEuEcWfewf8'

        Returns
        -------
        StringValue
            Extracted string value
        """
        import ibis.expr.operations as ops

        return ops.ParseURL(self, extract, key).to_expr()

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
        import ibis.expr.operations as ops

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
        import ibis.expr.operations as ops

        return ops.StringConcat([self, other, *args]).to_expr()

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
        return self.concat(other)

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
        import ibis.expr.rules as rlz

        return rlz.string(other).concat(self)

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
        import ibis.expr.operations as ops

        return ops.BaseConvert(self, from_base, to_base).to_expr()

    def __mul__(
        self, n: int | ir.IntegerValue
    ) -> StringValue | NotImplemented:
        import ibis.expr.operations as ops

        return _binop(ops.Repeat, self, n)

    __rmul__ = __mul__


@public
class StringScalar(Scalar, StringValue):
    pass


@public
class StringColumn(Column, StringValue):
    pass
