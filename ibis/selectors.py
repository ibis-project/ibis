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
('a', 'abcd')

Compare that to the [`numeric`](#ibis.selectors.numeric) selector:

>>> import ibis.selectors as s
>>> expr = t.select(s.numeric())
>>> expr.columns
('a', 'abcd')

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
('a', 'b', 'abcd')

Using a composition of selectors this is much less tiresome:

>>> expr = t.select((s.numeric() | s.of_type("string")) & s.contains(("a", "b", "cd")))
>>> expr.columns
('a', 'b', 'abcd')
"""

from __future__ import annotations

import builtins
import inspect
import operator
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from functools import reduce
from typing import Optional, Union

from public import public

import ibis.common.exceptions as exc
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.types as ir
from ibis import util
from ibis.common.collections import frozendict  # noqa: TC001
from ibis.common.deferred import Deferred, Resolver
from ibis.common.grounds import Concrete, Singleton
from ibis.common.selectors import All, Any, Expandable, Selector
from ibis.common.typing import VarTuple  # noqa: TC001


def __getattr__(name):
    if name == "c":
        util.warn_deprecated(
            "c", instead="use `ibis.selectors.cols` instead", as_of="9.5"
        )
        return cols
    elif name == "r":
        util.warn_deprecated(
            "r", instead="use `ibis.selectors.index` instead", as_of="9.5"
        )
        return index
    raise AttributeError(name)


class Where(Selector):
    predicate: Callable[[ir.Value], bool]

    def expand_names(self, table: ir.Table) -> frozenset[str]:
        predicate = self.predicate
        return frozenset(col for col in table.columns if predicate(table[col]))


@public
def where(predicate: Callable[[ir.Value], bool]) -> Selector:
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
    ('a',)

    """
    return Where(predicate)


@public
def numeric() -> Selector:
    """Return numeric columns.

    Examples
    --------
    >>> import ibis
    >>> import ibis.selectors as s
    >>> t = ibis.table(dict(a="int", b="string", c="array<string>"), name="t")
    >>> t.columns
    ('a', 'b', 'c')
    >>> expr = t.select(s.numeric())  # `a` has integer type, so it's numeric
    >>> expr.columns
    ('a',)

    See Also
    --------
    [`of_type`](#ibis.selectors.of_type)

    """
    return of_type(dt.Numeric)


class OfType(Selector):
    predicate: Callable[[dt.DataType], bool]

    def expand_names(self, table: ir.Table) -> frozenset[str]:
        predicate = self.predicate
        return frozenset(name for name, typ in table.schema().items() if predicate(typ))


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

    >>> import ibis
    >>> import ibis.expr.datatypes as dt
    >>> import ibis.selectors as s
    >>> t = ibis.table(dict(name="string", siblings="array<string>", parents="array<int64>"))
    >>> expr = t.select(s.of_type(dt.Array(dt.string)))
    >>> expr.columns
    ('siblings',)

    Strings are also accepted

    >>> expr = t.select(s.of_type("array<string>"))
    >>> expr.columns
    ('siblings',)

    Abstract/unparametrized types may also be specified by their string name
    (e.g. "integer" for any integer type), or by passing in a `DataType` class
    instead. The following options are equivalent.

    >>> expr1 = t.select(s.of_type("array"))
    >>> expr2 = t.select(s.of_type(dt.Array))
    >>> expr1.equals(expr2)
    True
    >>> expr2.columns
    ('siblings', 'parents')

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

        if dtype_cls := abstract.get(dtype.lower()):
            predicate = lambda typ, dtype_cls=dtype_cls: isinstance(typ, dtype_cls)
        else:
            dtype = dt.dtype(dtype)
            predicate = lambda typ, dtype=dtype: typ == dtype

    elif inspect.isclass(dtype) and issubclass(dtype, dt.DataType):
        predicate = lambda typ, dtype_cls=dtype: isinstance(typ, dtype_cls)
    else:
        dtype = dt.dtype(dtype)
        predicate = lambda typ, dtype=dtype: typ == dtype

    return OfType(predicate)


class StartsWith(Selector):
    prefixes: str | VarTuple[str]

    def expand_names(self, table: ir.Table) -> frozenset[str]:
        prefixes = self.prefixes
        return frozenset(col for col in table.columns if col.startswith(prefixes))


@public
def startswith(prefixes: str | tuple[str, ...]) -> Selector:
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
    ('apples', 'bananas')

    See Also
    --------
    [`endswith`](#ibis.selectors.endswith)

    """
    return StartsWith(prefixes)


class EndsWith(Selector):
    suffixes: str | VarTuple[str]

    def expand_names(self, table: ir.Table) -> frozenset[str]:
        suffixes = self.suffixes
        return frozenset(col for col in table.columns if col.endswith(suffixes))


@public
def endswith(suffixes: str | tuple[str, ...]) -> Selector:
    """Select columns whose name ends with one of `suffixes`.

    Parameters
    ----------
    suffixes
        Suffixes to compare column names against

    Examples
    --------
    >>> import ibis
    >>> import ibis.selectors as s
    >>> t = ibis.table(dict(model_id="int", model_name="str", user_id="int"), name="t")
    >>> expr = t.select(s.endswith("id"))
    >>> expr.columns
    ('model_id', 'user_id')

    See Also
    --------
    [`startswith`](#ibis.selectors.startswith)

    """
    return EndsWith(suffixes)


class Contains(Selector):
    needles: VarTuple[str]
    how: Callable[[Iterable[bool]], bool]

    def expand_names(self, table: ir.Table) -> frozenset[str]:
        needles = self.needles
        how = self.how
        return frozenset(
            col for col in table.columns if how(map(col.__contains__, needles))
        )


@public
def contains(
    needles: str | tuple[str, ...],
    how: Callable[[Iterable[bool]], bool] = builtins.any,
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

    >>> import ibis
    >>> import ibis.selectors as s
    >>> t = ibis.table(
    ...     dict(a="int64", b="string", c="float", d="array<int16>", ab="struct<x: int>")
    ... )
    >>> expr = t.select(s.contains(("a", "b")))
    >>> expr.columns
    ('a', 'b', 'ab')

    Select columns that contain all of `"a"` and `"b"`, that is, both `"a"` and
    `"b"` must be in each column's name to match.

    >>> expr = t.select(s.contains(("a", "b"), how=all))
    >>> expr.columns
    ('ab',)

    See Also
    --------
    [`matches`](#ibis.selectors.matches)

    """
    return Contains(tuple(util.promote_list(needles)), how=how)


class Matches(Selector):
    regex: re.Pattern

    def expand_names(self, table: ir.Table) -> frozenset[str]:
        return frozenset(filter(self.regex.search, table.columns))


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
    ('ab', 'abd')

    See Also
    --------
    [`contains`](#ibis.selectors.contains)

    """
    return Matches(re.compile(regex))


@public
def any_of(*predicates: str | Selector) -> Selector:
    """Include columns satisfying any of `predicates`.

    Examples
    --------
    >>> import ibis
    >>> import ibis.selectors as s
    >>> t = ibis.table(dict(model_id="int", model_name="str", user_id="int"), name="t")
    >>> expr = t.select(s.any_of(s.endswith("id"), s.startswith("m")))
    >>> expr.columns
    ('model_id', 'model_name', 'user_id')
    """
    return Any(tuple(map(_to_selector, predicates)))


@public
def all_of(*predicates: str | Selector) -> Selector:
    """Include columns satisfying all of `predicates`.

    Examples
    --------
    >>> import ibis
    >>> import ibis.selectors as s
    >>> t = ibis.table(dict(model_id="int", model_name="str", user_id="int"), name="t")
    >>> expr = t.select(s.all_of(s.endswith("id"), s.startswith("m")))
    >>> expr.columns
    ('model_id',)
    """
    return All(tuple(map(_to_selector, predicates)))


class Cols(Selector):
    names: frozenset[str]

    def expand_names(self, table: ir.Table) -> frozenset[str]:
        names = self.names
        columns = table.columns
        if extra_cols := sorted(names.difference(columns)):
            raise exc.IbisInputError(
                f"Columns {extra_cols} are not present in {columns}"
            )
        return names


@public
def cols(*names: str | ir.Column) -> Selector:
    """Select specific column names.

    Parameters
    ----------
    names
        The column names to select

    Examples
    --------
    >>> import ibis
    >>> import ibis.selectors as s
    >>> t = ibis.table({"a": "int", "b": "int", "c": "int"})
    >>> expr = t.select(s.cols("a", "b"))
    >>> expr.columns
    ('a', 'b')

    See Also
    --------
    [`index`](#ibis.selectors.cols)
    """
    names = frozenset(col if isinstance(col, str) else col.get_name() for col in names)
    return Cols(names)


class Across(Concrete, Expandable):
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

                orig_name = orig_col.get_name()

                if callable(names):
                    name = names(orig_name, func_name)
                else:
                    name = names.format(col=orig_name, fn=func_name)

                if not isinstance(col.op(), ops.Alias):
                    col = col.name(name)

                expanded.append(col)

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
    selector = _to_selector(selector)
    return Across(selector=selector, funcs=funcs, names=names)


class IfAnyAll(Concrete, Expandable):
    selector: Selector
    predicate: Union[Resolver, Callable[[ir.Value], ir.BooleanValue]]
    summarizer: Callable[[ir.BooleanValue, ir.BooleanValue], ir.BooleanValue]

    def expand(self, table: ir.Table) -> Sequence[ir.Value]:
        func = self.predicate

        if isinstance(func, Resolver):
            fn = lambda col, func=func: func.resolve({"_": col})
        else:
            fn = func

        return [reduce(self.summarizer, map(fn, self.selector.expand(table)))]


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


class Slice(Concrete):
    """Hashable and smaller-scoped slice object versus the builtin one."""

    start: int | str | None = None
    stop: int | str | None = None
    step: int | None = None


class ColumnIndex(Selector):
    key: str | int | Slice | VarTuple[int | str]

    @staticmethod
    def slice_key_to_int(
        value: int | str | None, name_locs: Mapping[str, int], offset: int
    ) -> int:
        if value is None or isinstance(value, int):
            return value
        else:
            assert isinstance(value, str), f"expected `str` got {type(value)}"
            return name_locs[value] + offset

    def expand_names(self, table: ir.Table) -> frozenset[str]:
        name_locs = table.schema()._name_locs
        key = self.key

        if isinstance(key, str):
            iterable = (key,)
        elif isinstance(key, int):
            iterable = (table.columns[key],)
        elif isinstance(key, Slice):
            start = self.slice_key_to_int(key.start, name_locs, offset=0)
            stop = self.slice_key_to_int(key.stop, name_locs, offset=1)
            step = key.step
            iterable = table.columns[start:stop:step]
        else:
            iterable = (
                table.columns[el if isinstance(el, int) else name_locs[el]]
                for el in key
            )
        return frozenset(iterable)


class Indexable(Singleton):
    def __getitem__(self, key: str | int | slice | Iterable[int | str]):
        if isinstance(key, slice):
            key = Slice(key.start, key.stop, key.step)
        return ColumnIndex(key)


index = Indexable()
"""Select columns by index.

Examples
--------
>>> import ibis
>>> import ibis.selectors as s
>>> t = ibis.table(
...     {"a": "int", "b": "int", "c": "int", "d": "int", "e": "int"}
... )

Select one column by numeric index:
>>> expr = t.select(s.index[0])
>>> expr.columns
['a']

Select multiple columns by numeric index:
>>> expr = t.select(s.index[[0, 1]])
>>> expr.columns
['a', 'b']

Select a slice of columns by numeric index:
>>> expr = t.select(s.index[1:4])
>>> expr.columns
['b', 'c', 'd']

Select a slice of columns by name:
>>> expr = t.select(s.index["b":"d"])
>>> expr.columns
['b', 'c', 'd']

See Also
--------
[`cols`](#ibis.selectors.cols)
"""


class First(Singleton, Selector):
    def expand(self, table: ir.Table) -> Sequence[ir.Value]:
        return [table[0]]

    def expand_names(self, table: ir.Table) -> frozenset[str]:
        return frozenset((table.columns[0],))


@public
def first() -> Selector:
    """Return the first column of a table.

    Examples
    --------
    >>> import ibis
    >>> import ibis.selectors as s
    >>> t = ibis.table(dict(model_id="int", model_name="str", user_id="int"), name="t")
    >>> expr = t.select(s.first())
    >>> expr.columns
    ('model_id',)
    """
    return First()


class Last(Singleton, Selector):
    def expand(self, table: ir.Table) -> Sequence[ir.Value]:
        return [table[-1]]

    def expand_names(self, table: ir.Table) -> frozenset[str]:
        return frozenset((table.columns[-1],))


@public
def last() -> Selector:
    """Return the last column of a table.

    Examples
    --------
    >>> import ibis
    >>> import ibis.selectors as s
    >>> t = ibis.table(dict(model_id="int", model_name="str", user_id="int"), name="t")
    >>> expr = t.select(s.last())
    >>> expr.columns
    ('user_id',)
    """
    return Last()


class AllColumns(Singleton, Selector):
    def expand(self, table: ir.Table) -> Sequence[ir.Value]:
        return list(map(table.__getitem__, table.columns))

    def expand_names(self, table: ir.Table) -> frozenset[str]:
        return frozenset(table.columns)


@public
def all() -> Selector:
    """Return every column from a table.

    Examples
    --------
    >>> import ibis
    >>> import ibis.selectors as s
    >>> t = ibis.table(dict(model_id="int", model_name="str", user_id="int"), name="t")
    >>> expr = t.select(s.all())
    >>> expr.columns
    ('model_id', 'model_name', 'user_id')
    """
    return AllColumns()


class NoColumns(Singleton, Selector):
    def expand(self, table: ir.Table) -> Sequence[ir.Value]:
        return []

    def expand_names(self, table: ir.Table) -> frozenset[str]:
        return frozenset()


@public
def none() -> Selector:
    """Return no columns.

    Examples
    --------
    >>> import ibis
    >>> import ibis.selectors as s
    >>> ibis.options.interactive = True
    >>> t = ibis.memtable(
    ...     {
    ...         "id": [1, 2, 3, 4, 5, 6],
    ...         "color": ["Red", "Green", "Blue", "Blue", "Red", "Blue"],
    ...     }
    ... )

    `s.none()` results in an empty expansion.

    >>> s.none().expand(t)
    []

    This can be useful when you want to pivot a table without identifying unique
    observations.

    >>> t.pivot_wider(
    ...     id_cols=s.none(),
    ...     names_from="color",
    ...     values_from="color",
    ...     values_agg="count",
    ...     names_sort=True,
    ... )
    ┏━━━━━━━┳━━━━━━━┳━━━━━━━┓
    ┃ Blue  ┃ Green ┃ Red   ┃
    ┡━━━━━━━╇━━━━━━━╇━━━━━━━┩
    │ int64 │ int64 │ int64 │
    ├───────┼───────┼───────┤
    │     3 │     1 │     2 │
    └───────┴───────┴───────┘
    """
    return NoColumns()


def _to_selector(
    obj: str | Selector | ir.Column | Sequence[str | Selector | ir.Column],
) -> Selector:
    """Convert an object to a `Selector`."""
    if isinstance(obj, Selector):
        return obj
    elif isinstance(obj, ir.Column):
        return cols(obj.get_name())
    elif isinstance(obj, str):
        return cols(obj)
    elif isinstance(obj, Expandable):
        raise exc.IbisInputError(
            f"Cannot compose {obj.__class__.__name__} with other selectors"
        )
    elif not obj:
        return none()
    else:
        return any_of(*obj)
