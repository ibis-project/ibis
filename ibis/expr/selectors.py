"""Construct column selectors."""

from __future__ import annotations

import re
from typing import Callable, Iterable, Sequence

import ibis.expr.datatypes as dt
import ibis.expr.types as ir
from ibis import util


class Selector:
    __slots__ = ("predicate",)

    def __init__(self, predicate: Callable[[ir.Column], bool]) -> None:
        """Construct a `Selector` with `predicate`."""
        self.predicate = predicate

    def expand(self, table: ir.Table) -> Sequence[ir.Column]:
        """Evaluate `self.predicate` on every column of `table`."""
        return [col for column in table.columns if self.predicate(col := table[column])]

    def __and__(self, other: Selector) -> Selector:
        """Compute the conjunction of two `Selectors`."""
        return self.__class__(lambda col: self.predicate(col) and other.predicate(col))

    def __or__(self, other: Selector) -> Selector:
        """Compute the disjunction of two `Selectors`."""
        return self.__class__(lambda col: self.predicate(col) or other.predicate(col))

    def __invert__(self) -> Selector:
        """Compute the logical negation of two `Selectors`."""
        return self.__class__(lambda col: not self.predicate(col))


def where(predicate: Callable[[ir.Value], bool]) -> Selector:
    """Return columns that satisfy `predicate`.

    Examples
    --------
    >>> t = ibis.table(dict(a="float32"), name="t")
    >>> t.select(s.where(lambda col: col.get_name() == "a"))
    r0 := UnboundTable: t
      a float32

    Selection[r0]
      selections:
        a: r0.a
    """
    return Selector(predicate)


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

    Selection[r0]
      selections:
        a: r0.a
    """
    return Selector(lambda col: col.type().is_numeric())


def of_type(dtype: dt.DataType | str | type[dt.DataType]) -> Selector:
    """Select columns of type `dtype`."""
    if isinstance(dtype, type):
        predicate = lambda col, dtype=dtype: isinstance(col.type(), dtype)
    else:
        dtype = dt.dtype(dtype)
        predicate = lambda col, dtype=dtype: col.type() == dtype
    return where(predicate)


def startswith(prefixes: str | tuple[str, ...]) -> Selector:
    """Select columns whose name starts with one of `prefixes`."""
    return where(lambda col, prefixes=prefixes: col.get_name().startswith(prefixes))


def endswith(suffixes: str | tuple[str, ...]) -> Selector:
    """Select columns whose name ends with one of `suffixes`."""
    return where(lambda col, suffixes=suffixes: col.get_name().endswith(suffixes))


def contains(
    needles: str | tuple[str, ...], how: Callable[[Iterable[bool]], bool] = any
) -> Selector:
    """Return columns whose name contains `needles`."""

    def predicate(
        col: ir.Column,
        needles: str | tuple[str, ...] = needles,
        how: Callable[[Iterable[bool]], bool] = how,
    ) -> bool:
        name = col.get_name()
        return how(needle in name for needle in util.promote_list(needles))

    return where(predicate)


def matches(regex: str | re.Pattern) -> Selector:
    """Return columns matching the regular expression `regex`."""
    pattern = re.compile(regex)
    return where(
        lambda col, pattern=pattern: pattern.search(col.get_name()) is not None
    )
