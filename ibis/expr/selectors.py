"""Convenient column selectors.

## Rationale

Column selectors are convenience functions for selecting columns that share some property.

## Discussion

For example, a common task is to be able to select all numeric columns for a
subsequent computation.

Without selectors this becomes quite verbose and tedious to write:

```python
>>> t.select([t[c] for c in t.columns if t[c].type().is_numeric()])  # doctest: +SKIP
```

Compare that to the [`numeric`][ibis.expr.selectors.numeric] selector:

```python
>>> t.select(s.numeric())  # doctest: +SKIP
```

When there are multiple properties to check it gets worse:

```python
>>> t.select(  # doctest: +SKIP
...     [
...         t[c] for c in t.columns
...         if t[c].type().is_numeric()
...         if ("a" in c.get_name() or "cd" in c.get_name())
...     ]
... )
```

Using a composition of selectors this is much less tiresome:

```python
>>> t.select(s.numeric() & s.contains(("a", "cd")))  # doctest: +SKIP
```
"""

from __future__ import annotations

import abc
import functools
import inspect
import operator
import re
from typing import Callable, Iterable, Mapping, Optional, Sequence, Union

from public import public

import ibis.expr.datatypes as dt
import ibis.expr.types as ir
from ibis import util
from ibis.common.annotations import attribute
from ibis.common.collections import frozendict
from ibis.common.grounds import Concrete, Singleton
from ibis.common.validators import Coercible
from ibis.expr.deferred import Deferred


class Selector(Concrete):
    """A column selector."""

    @abc.abstractmethod
    def expand(self, table: ir.Table) -> Sequence[ir.Value]:
        """Expand `table` into a sequence of value expressions.

        Parameters
        ----------
        table
            An ibis table expression

        Returns
        -------
        Sequence[Value]
            A sequence of value expressions
        """


class Predicate(Selector):
    predicate: Callable[[ir.Value], bool]

    def expand(self, table: ir.Table) -> Sequence[ir.Value]:
        """Evaluate `self.predicate` on every column of `table`.

        Parameters
        ----------
        table
            An ibis table expression
        """
        return [col for column in table.columns if self.predicate(col := table[column])]

    def __and__(self, other: Selector) -> Predicate:
        """Compute the conjunction of two `Selectors`.

        Parameters
        ----------
        other
            Another selector
        """
        return self.__class__(lambda col: self.predicate(col) and other.predicate(col))

    def __or__(self, other: Selector) -> Predicate:
        """Compute the disjunction of two `Selectors`.

        Parameters
        ----------
        other
            Another selector
        """
        return self.__class__(lambda col: self.predicate(col) or other.predicate(col))

    def __invert__(self) -> Predicate:
        """Compute the logical negation of two `Selectors`."""
        return self.__class__(lambda col: not self.predicate(col))


@public
def where(predicate: Callable[[ir.Value], bool]) -> Predicate:
    """Return columns that satisfy `predicate`.

    Use this selector when one of the other selectors does not meet your needs.

    Parameters
    ----------
    predicate
        A callable that accepts an ibis value expression and returns a `bool`

    Examples
    --------
    >>> import ibis
    >>> import ibis.expr.selectors as s
    >>> t = ibis.table(dict(a="float32"), name="t")
    >>> expr = t.select(s.where(lambda col: col.get_name() == "a"))
    >>> expr.columns
    ['a']
    """
    return Predicate(predicate=predicate)


@public
def numeric() -> Predicate:
    """Return numeric columns.

    Examples
    --------
    >>> import ibis
    >>> import ibis.expr.selectors as s
    >>> t = ibis.table(dict(a="int", b="string", c="array<string>"), name="t")
    >>> t
    UnboundTable: t
      a int64
      b string
      c array<string>
    >>> expr = t.select(s.numeric())  # `a` has integer type, so it's numeric
    >>> expr.columns
    ['a']

    See Also
    --------
    [`of_type`][ibis.expr.selectors.of_type]
    """
    return of_type(dt.Numeric)


@public
def of_type(dtype: dt.DataType | str | type[dt.DataType]) -> Predicate:
    """Select columns of type `dtype`.

    Parameters
    ----------
    dtype
        `DataType` instance, `str` or `DataType` class

    Examples
    --------
    Select according to a specific `DataType` instance

    >>> t.select(s.of_type(dt.Array(dt.string)))  # doctest: +SKIP

    Strings are also accepted

    >>> t.select(s.of_type("map<string, float>"))  # doctest: +SKIP

    Select by category of `DataType` by passing the `DataType` class

    >>> t.select(s.of_type(dt.Struct))  # doctest: +SKIP

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
def startswith(prefixes: str | tuple[str, ...]) -> Predicate:
    """Select columns whose name starts with one of `prefixes`.

    Parameters
    ----------
    prefixes
        Prefixes to compare column names against

    Examples
    --------
    >>> import ibis
    >>> import ibis.expr.selectors as s
    >>> t = ibis.table(dict(apples="int", oranges="float", bananas="bool"), name="t")
    >>> expr = t.select(s.startswith(("a", "b")))
    >>> expr.columns
    ['apples', 'bananas']

    See Also
    --------
    [`endswith`][ibis.expr.selectors.endswith]
    """
    return where(lambda col: col.get_name().startswith(prefixes))


@public
def endswith(suffixes: str | tuple[str, ...]) -> Predicate:
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
) -> Predicate:
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

    >>> t.select(s.contains(("a", "b")))  # doctest: +SKIP

    Select columns that contain all of `"a"` and `"b"`

    >>> t.select(s.contains(("a", "b"), how=all))  # doctest: +SKIP

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
    >>> t.select(s.matches(r"ab+"))  # doctest: +SKIP

    See Also
    --------
    [`contains`][ibis.expr.selectors.contains]
    """
    pattern = re.compile(regex)
    return where(lambda col: pattern.search(col.get_name()) is not None)


@public
def any_of(*predicates: Predicate) -> Predicate:
    """Include columns satisfying any of `predicates`."""
    return functools.reduce(operator.or_, predicates)


@public
def all_of(*predicates: Predicate) -> Predicate:
    """Include columns satisfying all of `predicates`."""
    return functools.reduce(operator.and_, predicates)


@public
def c(*names: str) -> Predicate:
    """Select specific column names."""
    names = frozenset(names)
    return where(lambda col: col.get_name() in names)


class Across(Selector):
    selector: Selector
    funcs: Union[
        Deferred,
        Callable[[ir.Value], ir.Value],
        frozendict[Optional[str], Union[Deferred, Callable[[ir.Value], ir.Value]]],
    ]
    names: Union[str, Callable[[str, Optional[str]], str]]

    def expand(self, table: ir.Table) -> Sequence[ir.Value]:
        expanded = []

        names = self.names
        cols = self.selector.expand(table)
        for func_name, func in self.funcs.items():
            resolve = func.resolve if isinstance(func, Deferred) else func
            expanded.extend(
                col.name(
                    names(orig_col.get_name(), func_name)
                    if callable(names)
                    else names.format(col=orig_col.get_name(), fn=func_name)
                )
                for col, orig_col in zip(map(resolve, cols), cols)
            )

        return expanded


@public
def across(selector, func, names=None) -> Across:
    if names is None:
        names = lambda col, fn: "_".join(filter(None, (col, fn)))
    funcs = frozendict(func if isinstance(func, Mapping) else {None: func})
    return Across(selector=selector, funcs=funcs, names=names)


class IfAnyAll(Selector):
    selector: Selector
    predicate: Union[Deferred, Callable[[ir.Value], ir.BooleanValue]]
    summarizer: Callable[[ir.BooleanValue, ir.BooleanValue], ir.BooleanValue]

    def expand(self, table: ir.Table) -> Sequence[ir.Value]:
        func = self.predicate
        resolve = func.resolve if isinstance(func, Deferred) else func
        return [
            functools.reduce(self.summarizer, map(resolve, self.selector.expand(table)))
        ]


@public
def if_any(selector: Selector, predicate: Deferred | Callable) -> IfAnyAll:
    return IfAnyAll(selector=selector, predicate=predicate, summarizer=operator.or_)


@public
def if_all(selector: Selector, predicate: Deferred | Callable) -> IfAnyAll:
    return IfAnyAll(selector=selector, predicate=predicate, summarizer=operator.and_)


class HashableSlice(Concrete, Coercible):
    slice: slice

    @classmethod
    def __coerce__(cls, slice):
        return cls(slice)

    @property
    def start(self):
        return self.slice.start

    @property
    def stop(self):
        return self.slice.stop

    @property
    def step(self):
        return self.slice.step

    @attribute.default
    def __precomputed_hash__(self) -> int:
        return hash((self.__class__, (self.start, self.stop, self.step)))


class Sliceable(Singleton):
    def __getitem__(self, key: str | int | slice | Iterable[int | str]) -> Predicate:
        import ibis.expr.analysis as an

        def pred(col: ir.Value) -> bool:
            table = an.find_first_base_table(col.op())
            schema = table.schema
            idxs = schema._name_locs
            num_names = len(schema)
            colname = col.get_name()
            colidx = idxs[colname]

            if isinstance(key, str):
                return key == colname
            elif isinstance(key, int):
                return key % num_names == colidx
            elif util.is_iterable(key):
                return any(
                    (isinstance(el, int) and el % num_names == colidx)
                    or (isinstance(el, str) and el == colname)
                    for el in key
                )
            else:
                start = key.start or 0
                stop = key.stop or num_names
                step = key.step or 1

                if isinstance(start, str):
                    start = idxs[start]

                if isinstance(stop, str):
                    stop = idxs[stop] + 1

                return colidx in range(start, stop, step)

        return where(pred)


r = Sliceable()


@public
def first() -> Predicate:
    """Return the first column of a table."""
    return r[0]


@public
def last() -> Predicate:
    """Return the last column of a table."""
    return r[-1]


@public
def all() -> Predicate:
    """Return every column from a table."""
    return r[:]
