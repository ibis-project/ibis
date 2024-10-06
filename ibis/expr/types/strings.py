from __future__ import annotations

import functools
import operator
from typing import TYPE_CHECKING, Any, Literal

from public import public

import ibis.expr.operations as ops
from ibis import util
from ibis.expr.types.core import _binop
from ibis.expr.types.generic import Column, Scalar, Value
from ibis.util import deprecated

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    import ibis.expr.types as ir


@public
class StringValue(Value):
    def __getitem__(self, key: slice | int | ir.IntegerScalar) -> StringValue:
        """Index or slice a string expression.

        Parameters
        ----------
        key
            [](`int`), [](`slice`) or integer scalar expression

        Returns
        -------
        StringValue
            Indexed or sliced string value

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable({"food": ["bread", "cheese", "rice"], "idx": [1, 2, 4]})
        >>> t
        ┏━━━━━━━━┳━━━━━━━┓
        ┃ food   ┃ idx   ┃
        ┡━━━━━━━━╇━━━━━━━┩
        │ string │ int64 │
        ├────────┼───────┤
        │ bread  │     1 │
        │ cheese │     2 │
        │ rice   │     4 │
        └────────┴───────┘
        >>> t.food[0]
        ┏━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ Substring(food, 0, 1) ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━━┩
        │ string                │
        ├───────────────────────┤
        │ b                     │
        │ c                     │
        │ r                     │
        └───────────────────────┘
        >>> t.food[:3]
        ┏━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ StringSlice(food, 3) ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━┩
        │ string               │
        ├──────────────────────┤
        │ bre                  │
        │ che                  │
        │ ric                  │
        └──────────────────────┘
        >>> t.food[3:5]
        ┏━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ StringSlice(food, 3, 5) ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━━━━┩
        │ string                  │
        ├─────────────────────────┤
        │ ad                      │
        │ es                      │
        │ e                       │
        └─────────────────────────┘
        >>> t.food[7]
        ┏━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ Substring(food, 7, 1) ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━━┩
        │ string                │
        ├───────────────────────┤
        │ ~                     │
        │ ~                     │
        │ ~                     │
        └───────────────────────┘
        """
        from ibis.expr import types as ir

        if isinstance(key, slice):
            start, stop, step = key.start, key.stop, key.step

            if isinstance(step, ir.Expr) or (step is not None and step != 1):
                raise ValueError("Step can only be 1")
            if start is not None and not isinstance(start, ir.Expr) and start < 0:
                raise ValueError(
                    "Negative slicing not yet supported, got start value "
                    f"of {start:d}"
                )
            if stop is not None and not isinstance(stop, ir.Expr) and stop < 0:
                raise ValueError(
                    "Negative slicing not yet supported, got stop value " f"of {stop:d}"
                )
            if start is None and stop is None:
                return self
            return ops.StringSlice(self, start, stop).to_expr()
        elif isinstance(key, int):
            return self.substr(key, 1)
        raise NotImplementedError(f"string __getitem__[{key.__class__.__name__}]")

    def length(self) -> ir.IntegerValue:
        """Compute the length of a string.

        Returns
        -------
        IntegerValue
            The length of each string in the expression

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable({"s": ["aaa", "a", "aa"]})
        >>> t.s.length()
        ┏━━━━━━━━━━━━━━━━━┓
        ┃ StringLength(s) ┃
        ┡━━━━━━━━━━━━━━━━━┩
        │ int32           │
        ├─────────────────┤
        │               3 │
        │               1 │
        │               2 │
        └─────────────────┘
        """
        return ops.StringLength(self).to_expr()

    def lower(self) -> StringValue:
        """Convert string to all lowercase.

        Returns
        -------
        StringValue
            Lowercase string

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable({"s": ["AAA", "a", "AA"]})
        >>> t
        ┏━━━━━━━━┓
        ┃ s      ┃
        ┡━━━━━━━━┩
        │ string │
        ├────────┤
        │ AAA    │
        │ a      │
        │ AA     │
        └────────┘
        >>> t.s.lower()
        ┏━━━━━━━━━━━━━━┓
        ┃ Lowercase(s) ┃
        ┡━━━━━━━━━━━━━━┩
        │ string       │
        ├──────────────┤
        │ aaa          │
        │ a            │
        │ aa           │
        └──────────────┘
        """
        return ops.Lowercase(self).to_expr()

    def upper(self) -> StringValue:
        """Convert string to all uppercase.

        Returns
        -------
        StringValue
            Uppercase string

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable({"s": ["aaa", "A", "aa"]})
        >>> t
        ┏━━━━━━━━┓
        ┃ s      ┃
        ┡━━━━━━━━┩
        │ string │
        ├────────┤
        │ aaa    │
        │ A      │
        │ aa     │
        └────────┘
        >>> t.s.upper()
        ┏━━━━━━━━━━━━━━┓
        ┃ Uppercase(s) ┃
        ┡━━━━━━━━━━━━━━┩
        │ string       │
        ├──────────────┤
        │ AAA          │
        │ A            │
        │ AA           │
        └──────────────┘
        """
        return ops.Uppercase(self).to_expr()

    def reverse(self) -> StringValue:
        """Reverse the characters of a string.

        Returns
        -------
        StringValue
            Reversed string

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable({"s": ["abc", "def", "ghi"]})
        >>> t
        ┏━━━━━━━━┓
        ┃ s      ┃
        ┡━━━━━━━━┩
        │ string │
        ├────────┤
        │ abc    │
        │ def    │
        │ ghi    │
        └────────┘
        >>> t.s.reverse()
        ┏━━━━━━━━━━━━┓
        ┃ Reverse(s) ┃
        ┡━━━━━━━━━━━━┩
        │ string     │
        ├────────────┤
        │ cba        │
        │ fed        │
        │ ihg        │
        └────────────┘
        """
        return ops.Reverse(self).to_expr()

    def ascii_str(self) -> ir.IntegerValue:
        """Return the numeric ASCII code of the first character of a string.

        Returns
        -------
        IntegerValue
            ASCII code of the first character of the input

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable({"s": ["abc", "def", "ghi"]})
        >>> t.s.ascii_str()
        ┏━━━━━━━━━━━━━━━━┓
        ┃ StringAscii(s) ┃
        ┡━━━━━━━━━━━━━━━━┩
        │ int32          │
        ├────────────────┤
        │             97 │
        │            100 │
        │            103 │
        └────────────────┘
        """
        return ops.StringAscii(self).to_expr()

    def strip(self) -> StringValue:
        r"""Remove whitespace from left and right sides of a string.

        Returns
        -------
        StringValue
            Stripped string

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable({"s": ["\ta\t", "\nb\n", "\vc\t"]})
        >>> t
        ┏━━━━━━━━┓
        ┃ s      ┃
        ┡━━━━━━━━┩
        │ string │
        ├────────┤
        │ \ta\t  │
        │ \nb\n  │
        │ \vc\t  │
        └────────┘
        >>> t.s.strip()
        ┏━━━━━━━━━━┓
        ┃ Strip(s) ┃
        ┡━━━━━━━━━━┩
        │ string   │
        ├──────────┤
        │ a        │
        │ b        │
        │ c        │
        └──────────┘
        """
        return ops.Strip(self).to_expr()

    def lstrip(self) -> StringValue:
        r"""Remove whitespace from the left side of string.

        Returns
        -------
        StringValue
            Left-stripped string

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable({"s": ["\ta\t", "\nb\n", "\vc\t"]})
        >>> t
        ┏━━━━━━━━┓
        ┃ s      ┃
        ┡━━━━━━━━┩
        │ string │
        ├────────┤
        │ \ta\t  │
        │ \nb\n  │
        │ \vc\t  │
        └────────┘
        >>> t.s.lstrip()
        ┏━━━━━━━━━━━┓
        ┃ LStrip(s) ┃
        ┡━━━━━━━━━━━┩
        │ string    │
        ├───────────┤
        │ a\t       │
        │ b\n       │
        │ c\t       │
        └───────────┘
        """
        return ops.LStrip(self).to_expr()

    def rstrip(self) -> StringValue:
        r"""Remove whitespace from the right side of string.

        Returns
        -------
        StringValue
            Right-stripped string

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable({"s": ["\ta\t", "\nb\n", "\vc\t"]})
        >>> t
        ┏━━━━━━━━┓
        ┃ s      ┃
        ┡━━━━━━━━┩
        │ string │
        ├────────┤
        │ \ta\t  │
        │ \nb\n  │
        │ \vc\t  │
        └────────┘
        >>> t.s.rstrip()
        ┏━━━━━━━━━━━┓
        ┃ RStrip(s) ┃
        ┡━━━━━━━━━━━┩
        │ string    │
        ├───────────┤
        │ \ta       │
        │ \nb       │
        │ \vc       │
        └───────────┘
        """
        return ops.RStrip(self).to_expr()

    def capitalize(self) -> StringValue:
        """Uppercase the first letter, lowercase the rest.

        This API matches the semantics of the Python [](`str.capitalize`)
        method.

        Returns
        -------
        StringValue
            Capitalized string

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable({"s": ["aBC", " abc", "ab cd", None]})
        >>> t.s.capitalize()
        ┏━━━━━━━━━━━━━━━┓
        ┃ Capitalize(s) ┃
        ┡━━━━━━━━━━━━━━━┩
        │ string        │
        ├───────────────┤
        │ Abc           │
        │  abc          │
        │ Ab cd         │
        │ NULL          │
        └───────────────┘
        """
        return ops.Capitalize(self).to_expr()

    @util.deprecated(
        instead="use the `capitalize` method", as_of="9.0", removed_in="10.0"
    )
    def initcap(self) -> StringValue:
        """Deprecated. Use `capitalize` instead."""
        return self.capitalize()

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

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable({"s": ["bab", "ddd", "eaf"]})
        >>> t.s.contains("a")
        ┏━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ StringContains(s, 'a') ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━━━┩
        │ boolean                │
        ├────────────────────────┤
        │ True                   │
        │ False                  │
        │ True                   │
        └────────────────────────┘
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

    def hexdigest(
        self,
        how: Literal["md5", "sha1", "sha256", "sha512"] = "sha256",
    ) -> ir.StringValue:
        """Return the hash digest of the input as a hex encoded string.

        Parameters
        ----------
        how
            Hash algorithm to use

        Returns
        -------
        StringValue
            Hexadecimal representation of the hash as a string

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable({"species": ["Adelie", "Chinstrap", "Gentoo"]})
        >>> t.species.hexdigest()
        ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ HexDigest(species)                                           ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
        │ string                                                           │
        ├──────────────────────────────────────────────────────────────────┤
        │ a4d7d46b27480037bc1e513e0e157cbf258baae6ee69e3110d0f9ff418b57a3c │
        │ cb97d113ca69899ae4f1fb581f4a90d86989db77b4a33873d604b0ee412b4cc9 │
        │ b5e90cdff65949fe6bc226823245f7698110e563a12363fc57b3eed3e4a0a612 │
        └──────────────────────────────────────────────────────────────────┘
        """
        return ops.HexDigest(self, how.lower()).to_expr()

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

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable({"s": ["abc", "defg", "hijlk"]})
        >>> t.s.substr(2)
        ┏━━━━━━━━━━━━━━━━━┓
        ┃ Substring(s, 2) ┃
        ┡━━━━━━━━━━━━━━━━━┩
        │ string          │
        ├─────────────────┤
        │ c               │
        │ fg              │
        │ jlk             │
        └─────────────────┘
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

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable({"s": ["abc", "defg", "hijlk"]})
        >>> t.s.left(2)
        ┏━━━━━━━━━━━━━━━━━━━━┓
        ┃ Substring(s, 0, 2) ┃
        ┡━━━━━━━━━━━━━━━━━━━━┩
        │ string             │
        ├────────────────────┤
        │ ab                 │
        │ de                 │
        │ hi                 │
        └────────────────────┘
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

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable({"s": ["abc", "defg", "hijlk"]})
        >>> t.s.right(2)
        ┏━━━━━━━━━━━━━━━━┓
        ┃ StrRight(s, 2) ┃
        ┡━━━━━━━━━━━━━━━━┩
        │ string         │
        ├────────────────┤
        │ bc             │
        │ fg             │
        │ lk             │
        └────────────────┘
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

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable({"s": ["a", "bb", "c"]})
        >>> t.s.repeat(5)
        ┏━━━━━━━━━━━━━━┓
        ┃ Repeat(s, 5) ┃
        ┡━━━━━━━━━━━━━━┩
        │ string       │
        ├──────────────┤
        │ aaaaa        │
        │ bbbbbbbbbb   │
        │ ccccc        │
        └──────────────┘
        """
        return ops.Repeat(self, n).to_expr()

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
        >>> table = ibis.table(dict(string_col="string"))
        >>> result = table.string_col.translate("a", "b")
        """
        return ops.Translate(self, from_str, to_str).to_expr()

    def find(
        self,
        substr: str | StringValue,
        start: int | ir.IntegerValue | None = None,
        end: int | ir.IntegerValue | None = None,
    ) -> ir.IntegerValue:
        """Return the position of the first occurrence of substring.

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

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable({"s": ["abc", "bac", "bca"]})
        >>> t.s.find("a")
        ┏━━━━━━━━━━━━━━━━━━━━┓
        ┃ StringFind(s, 'a') ┃
        ┡━━━━━━━━━━━━━━━━━━━━┩
        │ int64              │
        ├────────────────────┤
        │                  0 │
        │                  1 │
        │                  2 │
        └────────────────────┘
        >>> t.s.find("z")
        ┏━━━━━━━━━━━━━━━━━━━━┓
        ┃ StringFind(s, 'z') ┃
        ┡━━━━━━━━━━━━━━━━━━━━┩
        │ int64              │
        ├────────────────────┤
        │                 -1 │
        │                 -1 │
        │                 -1 │
        └────────────────────┘
        """
        if end is not None:
            raise NotImplementedError("`end` parameter is not yet implemented")
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
            Left-padded string

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable({"s": ["abc", "def", "ghij"]})
        >>> t.s.lpad(5, "-")
        ┏━━━━━━━━━━━━━━━━━┓
        ┃ LPad(s, 5, '-') ┃
        ┡━━━━━━━━━━━━━━━━━┩
        │ string          │
        ├─────────────────┤
        │ --abc           │
        │ --def           │
        │ -ghij           │
        └─────────────────┘
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

        Returns
        -------
        StringValue
            Right-padded string

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable({"s": ["abc", "def", "ghij"]})
        >>> t.s.rpad(5, "-")
        ┏━━━━━━━━━━━━━━━━━┓
        ┃ RPad(s, 5, '-') ┃
        ┡━━━━━━━━━━━━━━━━━┩
        │ string          │
        ├─────────────────┤
        │ abc--           │
        │ def--           │
        │ ghij-           │
        └─────────────────┘
        """
        return ops.RPad(self, length, pad).to_expr()

    def find_in_set(self, str_list: Sequence[str]) -> ir.IntegerValue:
        """Find the first occurrence of `str_list` within a list of strings.

        No string in `str_list` can have a comma.

        Parameters
        ----------
        str_list
            Sequence of strings

        Returns
        -------
        IntegerValue
            Position of `str_list` in `self`. Returns -1 if `self` isn't found
            or if `self` contains `','`.

        Examples
        --------
        >>> import ibis
        >>> table = ibis.table(dict(string_col="string"))
        >>> result = table.string_col.find_in_set(["a", "b"])
        """
        return ops.FindInSet(self, str_list).to_expr()

    def join(self, strings: Sequence[str | StringValue] | ir.ArrayValue) -> StringValue:
        """Join a list of strings using `self` as the separator.

        Parameters
        ----------
        strings
            Strings to join with `arg`

        Returns
        -------
        StringValue
            Joined string

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable({"arr": [["a", "b", "c"], None, [], ["b", None]]})
        >>> t
        ┏━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ arr                  ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━┩
        │ array<string>        │
        ├──────────────────────┤
        │ ['a', 'b', ... +1]   │
        │ NULL                 │
        │ []                   │
        │ ['b', None]          │
        └──────────────────────┘
        >>> ibis.literal("|").join(t.arr)
        ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ ArrayStringJoin(arr, '|') ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
        │ string                    │
        ├───────────────────────────┤
        │ a|b|c                     │
        │ NULL                      │
        │ NULL                      │
        │ b                         │
        └───────────────────────────┘

        See Also
        --------
        [`ArrayValue.join`](./expression-collections.qmd#ibis.expr.types.arrays.ArrayValue.join)
        """
        import ibis.expr.types as ir

        if isinstance(strings, ir.ArrayValue):
            cls = ops.ArrayStringJoin
        else:
            cls = ops.StringJoin
        return cls(strings, sep=self).to_expr()

    def startswith(self, start: str | StringValue) -> ir.BooleanValue:
        """Determine whether `self` starts with `start`.

        Parameters
        ----------
        start
            prefix to check for

        Returns
        -------
        BooleanValue
            Boolean indicating whether `self` starts with `start`

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable({"s": ["Ibis project", "GitHub"]})
        >>> t.s.startswith("Ibis")
        ┏━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ StartsWith(s, 'Ibis') ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━━┩
        │ boolean               │
        ├───────────────────────┤
        │ True                  │
        │ False                 │
        └───────────────────────┘
        """
        return ops.StartsWith(self, start).to_expr()

    def endswith(self, end: str | StringValue) -> ir.BooleanValue:
        """Determine if `self` ends with `end`.

        Parameters
        ----------
        end
            Suffix to check for

        Returns
        -------
        BooleanValue
            Boolean indicating whether `self` ends with `end`

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable({"s": ["Ibis project", "GitHub"]})
        >>> t.s.endswith("project")
        ┏━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ EndsWith(s, 'project') ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━━━┩
        │ boolean                │
        ├────────────────────────┤
        │ True                   │
        │ False                  │
        └────────────────────────┘
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

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable({"s": ["Ibis project", "GitHub"]})
        >>> t.s.like("%project")
        ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ StringSQLLike(s, '%project') ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
        │ boolean                      │
        ├──────────────────────────────┤
        │ True                         │
        │ False                        │
        └──────────────────────────────┘
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

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable({"s": ["Ibis project", "GitHub"]})
        >>> t.s.ilike("%PROJect")
        ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ StringSQLILike(s, '%PROJect') ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
        │ boolean                       │
        ├───────────────────────────────┤
        │ True                          │
        │ False                         │
        └───────────────────────────────┘
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

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable({"s": ["Ibis project", "GitHub"]})
        >>> t.s.re_search(".+Hub")
        ┏━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ RegexSearch(s, '.+Hub') ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━━━━┩
        │ boolean                 │
        ├─────────────────────────┤
        │ False                   │
        │ True                    │
        └─────────────────────────┘
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
            Regular expression pattern string
        index
            The index of the match group to return.

            The behavior of this function follows the behavior of Python's
            [`match objects`](https://docs.python.org/3/library/re.html#match-objects):
            when `index` is zero and there's a match, return the entire match,
            otherwise return the content of the `index`-th match group.

        Returns
        -------
        StringValue
            Extracted match or whole string if `index` is zero

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable({"s": ["abc", "bac", "bca"]})

        Extract a specific group

        >>> t.s.re_extract(r"^(a)bc", 1)
        ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ RegexExtract(s, '^(a)bc', 1) ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
        │ string                       │
        ├──────────────────────────────┤
        │ a                            │
        │ ~                            │
        │ ~                            │
        └──────────────────────────────┘

        Extract the entire match

        >>> t.s.re_extract(r"^(a)bc", 0)
        ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ RegexExtract(s, '^(a)bc', 0) ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
        │ string                       │
        ├──────────────────────────────┤
        │ abc                          │
        │ ~                            │
        │ ~                            │
        └──────────────────────────────┘
        """
        return ops.RegexExtract(self, pattern, index).to_expr()

    @util.backend_sensitive(
        why="Different backends support different regular expression syntax."
    )
    def re_split(self, pattern: str | StringValue) -> ir.ArrayValue:
        r"""Split a string by a regular expression `pattern`.

        Parameters
        ----------
        pattern
            Regular expression string to split by

        Returns
        -------
        ArrayValue
            Array of strings from splitting by `pattern`

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable(dict(s=["a.b", "b.....c", "c.........a", "def"]))
        >>> t.s
        ┏━━━━━━━━━━━━━┓
        ┃ s           ┃
        ┡━━━━━━━━━━━━━┩
        │ string      │
        ├─────────────┤
        │ a.b         │
        │ b.....c     │
        │ c.........a │
        │ def         │
        └─────────────┘
        >>> t.s.re_split(r"\.+").name("splits")
        ┏━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ splits               ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━┩
        │ array<string>        │
        ├──────────────────────┤
        │ ['a', 'b']           │
        │ ['b', 'c']           │
        │ ['c', 'a']           │
        │ ['def']              │
        └──────────────────────┘
        """
        return ops.RegexSplit(self, pattern).to_expr()

    @util.backend_sensitive(
        why="Different backends support different regular expression syntax."
    )
    def re_replace(
        self,
        pattern: str | StringValue,
        replacement: str | StringValue,
    ) -> StringValue:
        r"""Replace all matches found by regex `pattern` with `replacement`.

        Parameters
        ----------
        pattern
            Regular expression string
        replacement
            Replacement string or regular expression

        Returns
        -------
        StringValue
            Modified string

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable({"s": ["abc", "bac", "bca", "this has  multi \t whitespace"]})
        >>> s = t.s

        Replace all "a"s that are at the beginning of the string with "b":

        >>> s.re_replace("^a", "b")
        ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ RegexReplace(s, '^a', 'b')    ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
        │ string                        │
        ├───────────────────────────────┤
        │ bbc                           │
        │ bac                           │
        │ bca                           │
        │ this has  multi \t whitespace │
        └───────────────────────────────┘

        Double up any "a"s or "b"s, using capture groups and backreferences:

        >>> s.re_replace("([ab])", r"\0\0")
        ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ RegexReplace(s, '()', '\\0\\0')     ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
        │ string                              │
        ├─────────────────────────────────────┤
        │ aabbc                               │
        │ bbaac                               │
        │ bbcaa                               │
        │ this haas  multi \t whitespaace     │
        └─────────────────────────────────────┘

        Normalize all whitespace to a single space:

        >>> s.re_replace(r"\s+", " ")
        ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ RegexReplace(s, '\\s+', ' ') ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
        │ string                       │
        ├──────────────────────────────┤
        │ abc                          │
        │ bac                          │
        │ bca                          │
        │ this has multi whitespace    │
        └──────────────────────────────┘
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

        Returns
        -------
        StringValue
            Replaced string

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable({"s": ["abc", "bac", "bca"]})
        >>> t.s.replace("b", "z")
        ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ StringReplace(s, 'b', 'z') ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
        │ string                     │
        ├────────────────────────────┤
        │ azc                        │
        │ zac                        │
        │ zca                        │
        └────────────────────────────┘
        """
        return ops.StringReplace(self, pattern, replacement).to_expr()

    def as_timestamp(self, format_str: str) -> ir.TimestampValue:
        """Parse a string and return a timestamp.

        Parameters
        ----------
        format_str
            Format string in `strptime` format

        Returns
        -------
        TimestampValue
            Parsed timestamp value

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable({"ts": ["20170206"]})
        >>> t.ts.as_timestamp("%Y%m%d")
        ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ StringToTimestamp(ts, '%Y%m%d') ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
        │ timestamp('UTC')                │
        ├─────────────────────────────────┤
        │ 2017-02-06 00:00:00+00:00       │
        └─────────────────────────────────┘
        """
        return ops.StringToTimestamp(self, format_str).to_expr()

    @deprecated(as_of="10.0", instead="use as_timestamp() instead")
    def to_timestamp(self, format_str: str) -> ir.TimestampValue:
        return self.as_timestamp(format_str=format_str)

    def as_date(self, format_str: str) -> ir.DateValue:
        """Parse a string and return a date.

        Parameters
        ----------
        format_str
            Format string in `strptime` format

        Returns
        -------
        DateValue
            Parsed date value

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable({"ts": ["20170206"]})
        >>> t.ts.as_date("%Y%m%d")
        ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ StringToDate(ts, '%Y%m%d') ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
        │ date                       │
        ├────────────────────────────┤
        │ 2017-02-06                 │
        └────────────────────────────┘
        """
        return ops.StringToDate(self, format_str).to_expr()

    @deprecated(as_of="10.0", instead="use as_date() instead")
    def to_date(self, format_str: str) -> ir.DateValue:
        return self.as_date(format_str=format_str)

    def as_time(self, format_str: str) -> ir.TimeValue:
        """Parse a string and return a time.

        Parameters
        ----------
        format_str
            Format string in `strptime` format

        Returns
        -------
        TimeValue
            Parsed time value

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable({"ts": ["20:01:02"]})
        >>> t.ts.as_time("%H:%M:%S")
        ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ StringToTime(ts, '%H:%M:%S') ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
        │ time                         │
        ├──────────────────────────────┤
        │ 20:01:02                     │
        └──────────────────────────────┘
        """
        return ops.StringToTime(self, format_str).to_expr()

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
        >>> result = url.userinfo()  # user:pass

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
        >>> result = url.host()  # example.com

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
        >>> url = ibis.literal(
        ...     "https://example.com:80/docs/books/tutorial/index.html?name=networking"
        ... )
        >>> result = url.file()  # docs/books/tutorial/index.html?name=networking

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
        >>> url = ibis.literal(
        ...     "https://example.com:80/docs/books/tutorial/index.html?name=networking"
        ... )
        >>> result = url.path()  # docs/books/tutorial/index.html

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
        >>> url = ibis.literal(
        ...     "https://example.com:80/docs/books/tutorial/index.html?name=networking"
        ... )
        >>> result = url.query()  # name=networking
        >>> query_name = url.query("name")  # networking

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

        ::: {.callout-note}
        ## This API only works on backends with array support.
        :::

        Parameters
        ----------
        delimiter
            Value to split by

        Returns
        -------
        ArrayValue
            The string split by `delimiter`

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable({"col": ["a,b,c", "d,e", "f"]})
        >>> t
        ┏━━━━━━━━┓
        ┃ col    ┃
        ┡━━━━━━━━┩
        │ string │
        ├────────┤
        │ a,b,c  │
        │ d,e    │
        │ f      │
        └────────┘
        >>> t.col.split(",")
        ┏━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ StringSplit(col, ',') ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━━┩
        │ array<string>         │
        ├───────────────────────┤
        │ ['a', 'b', ... +1]    │
        │ ['d', 'e']            │
        │ ['f']                 │
        └───────────────────────┘
        """
        return ops.StringSplit(self, delimiter).to_expr()

    def concat(self, other: str | StringValue, *args: str | StringValue) -> StringValue:
        """Concatenate strings.

        NULLs are propagated. This methods is equivalent to using the `+` operator.

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

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable({"s": ["abc", None]})
        >>> t.s.concat("xyz", "123")
        ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ StringConcat((s, 'xyz', '123')) ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
        │ string                          │
        ├─────────────────────────────────┤
        │ abcxyz123                       │
        │ NULL                            │
        └─────────────────────────────────┘
        >>> t.s + "xyz"
        ┏━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ StringConcat((s, 'xyz')) ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━━━━━┩
        │ string                   │
        ├──────────────────────────┤
        │ abcxyz                   │
        │ NULL                     │
        └──────────────────────────┘
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

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable({"s": ["abc", "bac", "bca"]})
        >>> t
        ┏━━━━━━━━┓
        ┃ s      ┃
        ┡━━━━━━━━┩
        │ string │
        ├────────┤
        │ abc    │
        │ bac    │
        │ bca    │
        └────────┘
        >>> t.s + "z"
        ┏━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ StringConcat((s, 'z')) ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━━━┩
        │ string                 │
        ├────────────────────────┤
        │ abcz                   │
        │ bacz                   │
        │ bcaz                   │
        └────────────────────────┘
        >>> t.s + t.s
        ┏━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ StringConcat((s, s)) ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━┩
        │ string               │
        ├──────────────────────┤
        │ abcabc               │
        │ bacbac               │
        │ bcabca               │
        └──────────────────────┘
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

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable({"s": ["abc", "bac", "bca"]})
        >>> t
        ┏━━━━━━━━┓
        ┃ s      ┃
        ┡━━━━━━━━┩
        │ string │
        ├────────┤
        │ abc    │
        │ bac    │
        │ bca    │
        └────────┘
        >>> "z" + t.s
        ┏━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ StringConcat(('z', s)) ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━━━┩
        │ string                 │
        ├────────────────────────┤
        │ zabc                   │
        │ zbac                   │
        │ zbca                   │
        └────────────────────────┘
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

    def __mul__(self, n: int | ir.IntegerValue) -> StringValue:
        return _binop(ops.Repeat, self, n)

    __rmul__ = __mul__

    def levenshtein(self, other: StringValue) -> ir.IntegerValue:
        """Return the Levenshtein distance between two strings.

        Parameters
        ----------
        other
            String to compare to

        Returns
        -------
        IntegerValue
            The edit distance between the two strings

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> s = ibis.literal("kitten")
        >>> s.levenshtein("sitting")
        ┌───┐
        │ 3 │
        └───┘
        """
        return ops.Levenshtein(self, other).to_expr()


@public
class StringScalar(Scalar, StringValue):
    pass


@public
class StringColumn(Column, StringValue):
    def __getitem__(self, key: slice | int | ir.IntegerScalar) -> StringColumn:
        return StringValue.__getitem__(self, key)
