"""Convenient column selectors.

## Rationale

Column selectors are convenience functions for selecting columns that share some property.

## Discussion

For example, a common task is to be able to select all numeric columns for a
subsequent computation.

Without selectors this becomes quite verbose and tedious to write:

```python
>>> t.select([t[c] for c in t.columns if t[c].type().is_numeric()])
```

Compare that to the [`numeric`][ibis.expr.selectors.numeric] selector:

```python
>>> t.select(s.numeric())
```

When there are multiple properties to check it gets worse:

```python
>>> t.select(
...     [
...         t[c] for c in t.columns
...         if t[c].type().is_numeric()
...         if ("a" in c.get_name() or "cd" in c.get_name())
...     ]
... )
```

Using a composition of selectors this is much less tiresome:

```python
>>> t.select(s.numeric() & s.contains(("a", "cd")))
```
"""

from __future__ import annotations

import inspect
import re
from typing import Callable, Iterable, Sequence

from public import public

import ibis.expr.datatypes as dt
import ibis.expr.types as ir
from ibis import util


class Selector:
    """A column selector."""

    __slots__ = ("predicate",)

    def __init__(self, predicate: Callable[[ir.Value], bool]) -> None:
        """Construct a `Selector` with `predicate`.

        Parameters
        ----------
        predicate
            A callable that accepts an ibis value expression and returns a `bool`.
        """
        self.predicate = predicate

    def expand(self, table: ir.Table) -> Sequence[ir.Value]:
        """Evaluate `self.predicate` on every column of `table`.

        Parameters
        ----------
        table
            An ibis table expression
        """
        return [col for column in table.columns if self.predicate(col := table[column])]

    def __and__(self, other: Selector) -> Selector:
        """Compute the conjunction of two `Selectors`.

        Parameters
        ----------
        other
            Another selector
        """
        return self.__class__(lambda col: self.predicate(col) and other.predicate(col))

    def __or__(self, other: Selector) -> Selector:
        """Compute the disjunction of two `Selectors`.

        Parameters
        ----------
        other
            Another selector
        """
        return self.__class__(lambda col: self.predicate(col) or other.predicate(col))

    def __invert__(self) -> Selector:
        """Compute the logical negation of two `Selectors`."""
        return self.__class__(lambda col: not self.predicate(col))


@public
def where(predicate: Callable[[ir.Value], bool]) -> Selector:
    """Return columns that satisfy `predicate`.

    Use this selector when one of the other selectors does not meet your needs.

    Parameters
    ----------
    predicate
        A callable that accepts an ibis value expression and returns a `bool`

    Examples
    --------
    >>> t = ibis.table(dict(a="float32"), name="t")
    >>> t.select(s.where(lambda col: col.get_name() == "a"))
    r0 := UnboundTable: t
      a float32
    <BLANKLINE>
    Selection[r0]
      selections:
        a: r0.a
    """
    return Selector(predicate)


@public
def numeric() -> Selector:
    """Return numeric columns.

    Examples
    --------
    >>> import ibis.selectors as s
    >>> t = ibis.table(dict(a="int", b="string", c="array<string>"), name="t")
    >>> t
    r0 := UnboundTable: t
      a int64
      b string
      c array<string>
    >>> t.select(s.numeric())  # `a` has integer type, so it's numeric
    r0 := UnboundTable: t
      a int64
      b string
      c array<string>
    <BLANKLINE>
    Selection[r0]
      selections:
        a: r0.a

    See Also
    --------
    [`of_type`][ibis.expr.selectors.of_type]
    """
    return of_type(dt.Numeric)


@public
def of_type(dtype: dt.DataType | str | type[dt.DataType]) -> Selector:
    """Select columns of type `dtype`.

    Parameters
    ----------
    dtype
        `DataType` instance, `str` or `DataType` class

    Examples
    --------
    Select according to a specific `DataType` instance

    >>> t.select(s.of_type(dt.Array(dt.string)))

    Strings are also accepted

    >>> t.select(s.of_type("map<string, float>"))

    Select by category of `DataType` by passing the `DataType` class

    >>> t.select(s.of_type(dt.Struct))  # all struct columns, regardless of field types

    See Also
    --------
    [`numeric`][ibis.expr.selectors.numeric]
    """
    if inspect.isclass(dtype):
        predicate = lambda col: isinstance(col.type(), dtype)
    else:
        dtype = dt.dtype(dtype)
        predicate = lambda col: col.type() == dtype
    return where(predicate)


@public
def startswith(prefixes: str | tuple[str, ...]) -> Selector:
    """Select columns whose name starts with one of `prefixes`.

    Parameters
    ----------
    prefixes
        Prefixes to compare column names against

    Examples
    --------
    >>> t = ibis.table(dict(apples="int", oranges="float", bananas="bool"), name="t")
    >>> t.select(s.startswith(("a", "b")))

    See Also
    --------
    [`endswith`][ibis.expr.selectors.endswith]
    """
    return where(lambda col: col.get_name().startswith(prefixes))


@public
def endswith(suffixes: str | tuple[str, ...]) -> Selector:
    """Select columns whose name ends with one of `suffixes`.

    Parameters
    ----------
    suffixes
        Suffixes to compare column names against

    See Also
    --------
    [`startswith`][ibis.expr.selectors.startswith]
    """
    return where(lambda col: col.get_name().endswith(suffixes))


@public
def contains(
    needles: str | tuple[str, ...], how: Callable[[Iterable[bool]], bool] = any
) -> Selector:
    """Return columns whose name contains `needles`.

    Parameters
    ----------
    needles
        One or more strings to search for in column names
    how
        A boolean reduction to allow the configuration of how `needles` are summarized.

    Examples
    --------
    Select columns that contain either `"a"` or `"b"`

    >>> t.select(s.contains(("a", "b")))

    Select columns that contain all of `"a"` and `"b"`

    >>> t.select(s.contains(("a", "b"), how=all))

    See Also
    --------
    [`matches`][ibis.expr.selectors.matches]
    """

    def predicate(col: ir.Value) -> bool:
        name = col.get_name()
        return how(needle in name for needle in util.promote_list(needles))

    return where(predicate)


@public
def matches(regex: str | re.Pattern) -> Selector:
    """Return columns whose name matches the regular expression `regex`.

    Parameters
    ----------
    regex
        A string or `re.Pattern` object

    Examples
    --------
    >>> t.select(s.matches(r"ab+"))

    See Also
    --------
    [`contains`][ibis.expr.selectors.contains]
    """
    pattern = re.compile(regex)
    return where(lambda col: pattern.search(col.get_name()) is not None)
