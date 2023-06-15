"""Convenient column selectors.

::: {.callout-tip}
## Check out the [blog post on selectors](../posts/selectors) for examples!
:::

## Rationale

Column selectors are convenience functions for selecting columns that share some property.

## Discussion

For example, a common task is to be able to select all numeric columns for a
subsequent computation.

Without selectors this becomes quite verbose and tedious to write:

>>> import ibis
>>> t = ibis.table(dict(a="int", b="string", c="array<int>", abcd="float"))
>>> expr = t.select([t[c] for c in t.columns if t[c].type().is_numeric()])
>>> expr.columns
['a', 'abcd']

Compare that to the [`numeric`](#ibis.selectors.numeric) selector:

>>> import ibis.selectors as s
>>> expr = t.select(s.numeric())
>>> expr.columns
['a', 'abcd']

When there are multiple properties to check it gets worse:

>>> expr = t.select(
...     [
...         t[c]
...         for c in t.columns
...         if t[c].type().is_numeric() or t[c].type().is_string()
...         if ("a" in c or "b" in c or "cd" in c)
...     ]
... )
>>> expr.columns
['a', 'b', 'abcd']

Using a composition of selectors this is much less tiresome:

>>> expr = t.select((s.numeric() | s.of_type("string")) & s.contains(("a", "b", "cd")))
>>> expr.columns
['a', 'b', 'abcd']
"""

from __future__ import annotations

import abc
import functools
import inspect
import operator
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Callable, Optional, Union

from public import public

import ibis.common.exceptions as exc
import ibis.expr.datatypes as dt
import ibis.expr.types as ir
from ibis import util
from ibis.common.collections import frozendict  # noqa: TCH001
from ibis.common.deferred import Deferred, Resolver
from ibis.common.grounds import Concrete, Singleton


class Selector(Concrete):
    """A column selector."""

    @abc.abstractmethod
    def expand(self, table: ir.Table) -> Sequence[ir.Value]:
        """Expand `table` into value expressions that match the selector.

        Parameters
        ----------
        table
            An ibis table expression

        Returns
        -------
        Sequence[Value]
            A sequence of value expressions that match the selector
        """

    def positions(self, table: ir.Table) -> Sequence[int]:
        """Expand `table` into column indices that match the selector.

        Parameters
        ----------
        table
            An ibis table expression

        Returns
        -------
        Sequence[int]
            A sequence of column indices where the selector matches
        """
        raise NotImplementedError(
            f"`positions` doesn't make sense for {self.__class__.__name__} selector"
        )


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

    def positions(self, table: ir.Table) -> Sequence[int]:
        return [
            i for i, column in enumerate(table.columns) if self.predicate(table[column])
        ]

    def __and__(self, other: Selector) -> Predicate:
        """Compute the conjunction of two `Selector`s.

        Parameters
        ----------
        other
            Another selector
        """
        return self.__class__(lambda col: self.predicate(col) and other.predicate(col))

    def __or__(self, other: Selector) -> Predicate:
        """Compute the disjunction of two `Selector`s.

        Parameters
        ----------
        other
            Another selector
        """
        return self.__class__(lambda col: self.predicate(col) or other.predicate(col))

    def __invert__(self) -> Predicate:
        """Compute the logical negation of two `Selector`s."""
        return self.__class__(lambda col: not self.predicate(col))


@public
def where(predicate: Callable[[ir.Value], bool]) -> Predicate:
    """Select columns that satisfy `predicate`.

    Use this selector when one of the other selectors does not meet your needs.

    Parameters
    ----------
    predicate
        A callable that accepts an ibis value expression and returns a `bool`

    Examples
    --------
    >>> import ibis
    >>> import ibis.selectors as s
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
    >>> import ibis.selectors as s
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
    [`of_type`](#ibis.selectors.of_type)
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

    >>> import ibis
    >>> import ibis.expr.datatypes as dt
    >>> import ibis.selectors as s
    >>> t = ibis.table(
    ...     dict(name="string", siblings="array<string>", parents="array<int64>")
    ... )
    >>> expr = t.select(s.of_type(dt.Array(dt.string)))
    >>> expr.columns
    ['siblings']

    Strings are also accepted

    >>> expr = t.select(s.of_type("array<string>"))
    >>> expr.columns
    ['siblings']

    Abstract/unparametrized types may also be specified by their string name
    (e.g. "integer" for any integer type), or by passing in a `DataType` class
    instead. The following options are equivalent.

    >>> expr1 = t.select(s.of_type("array"))
    >>> expr2 = t.select(s.of_type(dt.Array))
    >>> expr1.equals(expr2)
    True
    >>> expr2.columns
    ['siblings', 'parents']

    See Also
    --------
    [`numeric`](#ibis.selectors.numeric)
    """
    if isinstance(dtype, str):
        # A mapping of abstract or parametric types, to allow selecting all
        # subclasses/parametrizations of these types, rather than only a
        # specific instance.
        abstract = {
            "array": dt.Array,
            "decimal": dt.Decimal,
            "floating": dt.Floating,
            "geospatial": dt.GeoSpatial,
            "integer": dt.Integer,
            "map": dt.Map,
            "numeric": dt.Numeric,
            "struct": dt.Struct,
            "temporal": dt.Temporal,
        }
        if cls := abstract.get(dtype.lower()):
            predicate = lambda col: isinstance(col.type(), cls)
        else:
            dtype = dt.dtype(dtype)
            predicate = lambda col: col.type() == dtype
    elif inspect.isclass(dtype):
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
    >>> import ibis.selectors as s
    >>> t = ibis.table(dict(apples="int", oranges="float", bananas="bool"), name="t")
    >>> expr = t.select(s.startswith(("a", "b")))
    >>> expr.columns
    ['apples', 'bananas']

    See Also
    --------
    [`endswith`](#ibis.selectors.endswith)
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
    [`startswith`](#ibis.selectors.startswith)
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

    >>> import ibis
    >>> import ibis.selectors as s
    >>> t = ibis.table(
    ...     dict(
    ...         a="int64", b="string", c="float", d="array<int16>", ab="struct<x: int>"
    ...     )
    ... )
    >>> expr = t.select(s.contains(("a", "b")))
    >>> expr.columns
    ['a', 'b', 'ab']

    Select columns that contain all of `"a"` and `"b"`, that is, both `"a"` and
    `"b"` must be in each column's name to match.

    >>> expr = t.select(s.contains(("a", "b"), how=all))
    >>> expr.columns
    ['ab']

    See Also
    --------
    [`matches`](#ibis.selectors.matches)
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
    >>> import ibis
    >>> import ibis.selectors as s
    >>> t = ibis.table(dict(ab="string", abd="int", be="array<string>"))
    >>> expr = t.select(s.matches(r"ab+"))
    >>> expr.columns
    ['ab', 'abd']

    See Also
    --------
    [`contains`](#ibis.selectors.contains)
    """
    pattern = re.compile(regex)
    return where(lambda col: pattern.search(col.get_name()) is not None)


@public
def any_of(*predicates: str | Predicate) -> Predicate:
    """Include columns satisfying any of `predicates`."""
    return functools.reduce(operator.or_, map(_to_selector, predicates))


@public
def all_of(*predicates: str | Predicate) -> Predicate:
    """Include columns satisfying all of `predicates`."""
    return functools.reduce(operator.and_, map(_to_selector, predicates))


@public
def c(*names: str | ir.Column) -> Predicate:
    """Select specific column names."""
    names = frozenset(col if isinstance(col, str) else col.get_name() for col in names)

    def func(col: ir.Value) -> bool:
        schema = col.op().table.schema
        if extra_cols := (names - schema.keys()):
            raise exc.IbisInputError(
                f"Columns {extra_cols} are not present in {schema.names}"
            )
        return col.get_name() in names

    return where(func)


class Across(Selector):
    selector: Selector
    funcs: Union[
        Resolver,
        Callable[[ir.Value], ir.Value],
        frozendict[Optional[str], Union[Resolver, Callable[[ir.Value], ir.Value]]],
    ]
    names: Union[str, Callable[[str, Optional[str]], str]]

    def expand(self, table: ir.Table) -> Sequence[ir.Value]:
        expanded = []

        names = self.names
        cols = self.selector.expand(table)
        for func_name, func in self.funcs.items():
            for orig_col in cols:
                if isinstance(func, Resolver):
                    col = func.resolve({"_": orig_col})
                else:
                    col = func(orig_col)

                if callable(names):
                    name = names(orig_col.get_name(), func_name)
                else:
                    name = names.format(col=orig_col.get_name(), fn=func_name)

                expanded.append(col.name(name))

        return expanded


@public
def across(
    selector: Selector | Iterable[str] | str,
    func: Deferred
    | Callable[[ir.Value], ir.Value]
    | Mapping[str | None, Deferred | Callable[[ir.Value], ir.Value]],
    names: str | Callable[[str, str | None], str] | None = None,
) -> Across:
    """Apply data transformations across multiple columns.

    Parameters
    ----------
    selector
        An expression that selects columns on which the transformation function
        will be applied, an iterable of `str` column names or a single `str`
        column name.
    func
        A function (or dictionary of functions) to use to transform the data.
    names
        A lambda function or a format string to name the columns created by the
        transformation function.

    Returns
    -------
    Across
        An `Across` selector object

    Examples
    --------
    >>> import ibis
    >>> ibis.options.interactive = True
    >>> from ibis import _, selectors as s
    >>> t = ibis.examples.penguins.fetch()
    >>> t.select(s.startswith("bill")).mutate(
    ...     s.across(s.numeric(), dict(centered=_ - _.mean()), names="{fn}_{col}")
    ... )
    ┏━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━┓
    ┃ bill_length_mm ┃ bill_depth_mm ┃ centered_bill_length_mm ┃ … ┃
    ┡━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━┩
    │ float64        │ float64       │ float64                 │ … │
    ├────────────────┼───────────────┼─────────────────────────┼───┤
    │           39.1 │          18.7 │                -4.82193 │ … │
    │           39.5 │          17.4 │                -4.42193 │ … │
    │           40.3 │          18.0 │                -3.62193 │ … │
    │           NULL │          NULL │                    NULL │ … │
    │           36.7 │          19.3 │                -7.22193 │ … │
    │           39.3 │          20.6 │                -4.62193 │ … │
    │           38.9 │          17.8 │                -5.02193 │ … │
    │           39.2 │          19.6 │                -4.72193 │ … │
    │           34.1 │          18.1 │                -9.82193 │ … │
    │           42.0 │          20.2 │                -1.92193 │ … │
    │              … │             … │                       … │ … │
    └────────────────┴───────────────┴─────────────────────────┴───┘
    """
    if names is None:
        names = lambda col, fn: "_".join(filter(None, (col, fn)))
    funcs = dict(func if isinstance(func, Mapping) else {None: func})
    if not isinstance(selector, Selector):
        selector = c(*util.promote_list(selector))
    return Across(selector=selector, funcs=funcs, names=names)


class IfAnyAll(Selector):
    selector: Selector
    predicate: Union[Resolver, Callable[[ir.Value], ir.BooleanValue]]
    summarizer: Callable[[ir.BooleanValue, ir.BooleanValue], ir.BooleanValue]

    def expand(self, table: ir.Table) -> Sequence[ir.Value]:
        func = self.predicate
        if isinstance(func, Resolver):
            elems = (func.resolve({"_": col}) for col in self.selector.expand(table))
        else:
            elems = (func(col) for col in self.selector.expand(table))

        return [functools.reduce(self.summarizer, elems)]


@public
def if_any(selector: Selector, predicate: Deferred | Callable) -> IfAnyAll:
    """Return the **disjunction** of `predicate` applied on all `selector` columns.

    Parameters
    ----------
    selector
        A column selector
    predicate
        A callable or deferred object defining a predicate to apply to each
        column from `selector`.

    Examples
    --------
    >>> import ibis
    >>> from ibis import selectors as s, _
    >>> ibis.options.interactive = True
    >>> penguins = ibis.examples.penguins.fetch()
    >>> cols = s.across(s.endswith("_mm"), (_ - _.mean()) / _.std())
    >>> expr = penguins.mutate(cols).filter(s.if_any(s.endswith("_mm"), _.abs() > 2))
    >>> expr_by_hand = penguins.mutate(cols).filter(
    ...     (_.bill_length_mm.abs() > 2)
    ...     | (_.bill_depth_mm.abs() > 2)
    ...     | (_.flipper_length_mm.abs() > 2)
    ... )
    >>> expr.equals(expr_by_hand)
    True
    >>> expr
    ┏━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━┓
    ┃ species ┃ island ┃ bill_length_mm ┃ bill_depth_mm ┃ flipper_length_mm ┃ … ┃
    ┡━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━┩
    │ string  │ string │ float64        │ float64       │ float64           │ … │
    ├─────────┼────────┼────────────────┼───────────────┼───────────────────┼───┤
    │ Adelie  │ Biscoe │      -1.103002 │      0.733662 │         -2.056307 │ … │
    │ Gentoo  │ Biscoe │       1.113285 │     -0.431017 │          2.068368 │ … │
    │ Gentoo  │ Biscoe │       2.871660 │     -0.076550 │          2.068368 │ … │
    │ Gentoo  │ Biscoe │       1.900890 │     -0.734846 │          2.139483 │ … │
    │ Gentoo  │ Biscoe │       1.076652 │     -0.177826 │          2.068368 │ … │
    │ Gentoo  │ Biscoe │       0.856855 │     -0.582932 │          2.068368 │ … │
    │ Gentoo  │ Biscoe │       1.497929 │     -0.076550 │          2.068368 │ … │
    │ Gentoo  │ Biscoe │       1.388031 │     -0.431017 │          2.068368 │ … │
    │ Gentoo  │ Biscoe │       2.047422 │     -0.582932 │          2.068368 │ … │
    │ Adelie  │ Dream  │      -2.165354 │     -0.836123 │         -0.918466 │ … │
    │ …       │ …      │              … │             … │                 … │ … │
    └─────────┴────────┴────────────────┴───────────────┴───────────────────┴───┘
    """
    return IfAnyAll(selector=selector, predicate=predicate, summarizer=operator.or_)


@public
def if_all(selector: Selector, predicate: Deferred | Callable) -> IfAnyAll:
    """Return the **conjunction** of `predicate` applied on all `selector` columns.

    Parameters
    ----------
    selector
        A column selector
    predicate
        A callable or deferred object defining a predicate to apply to each
        column from `selector`.

    Examples
    --------
    >>> import ibis
    >>> from ibis import selectors as s, _
    >>> ibis.options.interactive = True
    >>> penguins = ibis.examples.penguins.fetch()
    >>> cols = s.across(s.endswith("_mm"), (_ - _.mean()) / _.std())
    >>> expr = penguins.mutate(cols).filter(s.if_all(s.endswith("_mm"), _.abs() > 1))
    >>> expr_by_hand = penguins.mutate(cols).filter(
    ...     (_.bill_length_mm.abs() > 1)
    ...     & (_.bill_depth_mm.abs() > 1)
    ...     & (_.flipper_length_mm.abs() > 1)
    ... )
    >>> expr.equals(expr_by_hand)
    True
    >>> expr
    ┏━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━┓
    ┃ species ┃ island    ┃ bill_length_mm ┃ bill_depth_mm ┃ flipper_length_mm ┃ … ┃
    ┡━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━┩
    │ string  │ string    │ float64        │ float64       │ float64           │ … │
    ├─────────┼───────────┼────────────────┼───────────────┼───────────────────┼───┤
    │ Adelie  │ Dream     │      -1.157951 │      1.088129 │         -1.416272 │ … │
    │ Adelie  │ Torgersen │      -1.231217 │      1.138768 │         -1.202926 │ … │
    │ Gentoo  │ Biscoe    │       1.149917 │     -1.443781 │          1.214987 │ … │
    │ Gentoo  │ Biscoe    │       1.040019 │     -1.089314 │          1.072757 │ … │
    │ Gentoo  │ Biscoe    │       1.131601 │     -1.089314 │          1.712792 │ … │
    │ Gentoo  │ Biscoe    │       1.241499 │     -1.089314 │          1.570562 │ … │
    │ Gentoo  │ Biscoe    │       1.351398 │     -1.494420 │          1.214987 │ … │
    └─────────┴───────────┴────────────────┴───────────────┴───────────────────┴───┘
    """
    return IfAnyAll(selector=selector, predicate=predicate, summarizer=operator.and_)


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
"""Ranges of columns."""


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


def _to_selector(
    obj: str | Selector | ir.Column | Sequence[str | Selector | ir.Column],
) -> Selector:
    """Convert an object to a `Selector`."""
    if isinstance(obj, Selector):
        return obj
    elif isinstance(obj, ir.Column):
        return c(obj.get_name())
    elif isinstance(obj, str):
        return c(obj)
    else:
        return any_of(*obj)
