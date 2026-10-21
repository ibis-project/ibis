"""Test utilities."""

from __future__ import annotations

import datetime
import decimal
import numbers
import pickle
import re
from typing import TYPE_CHECKING

import pytest

import ibis
import ibis.expr.types as ir
from ibis import util

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import pandas as pd


def assert_equal(left, right):
    """Assert that two ibis objects are equal."""

    if util.all_of([left, right], ibis.Schema):
        assert left.equals(right), f"Comparing schemas: \n{left!r} !=\n{right!r}"
    else:
        assert left.equals(right), f"Objects unequal: \n{left!r}\nvs\n{right!r}"


def assert_pickle_roundtrip(obj):
    """Assert that an ibis object remains the same after pickling and
    unpickling."""
    loaded = pickle.loads(pickle.dumps(obj))
    if hasattr(obj, "equals"):
        assert obj.equals(loaded)
    else:
        assert obj == loaded


def schemas_eq(left: ir.Expr, right: ir.Expr) -> bool:
    left_schema = left.as_table().schema()
    right_schema = right.as_table().schema()
    return left_schema == right_schema


def assert_decompile_roundtrip(
    expr: ir.Expr,
    snapshot=None,
    eq: Callable[[ir.Expr, ir.Expr], bool] = ir.Expr.equals,
):
    """Assert that an ibis expression remains the same after decompilation.

    Parameters
    ----------
    expr
        The expression to decompile.
    snapshot
        A snapshot fixture.
    eq
        A callable that returns whether two Ibis expressions are equal.
        Defaults to `ibis.expr.types.Expr.equals`. Use this to adjust
        comparison behavior for expressions that contain `SelfReference`
        operations from table.view() calls, or other relations whose equality
        is difficult to roundtrip.
    """
    pytest.importorskip("black")

    rendered = ibis.decompile(expr, format=True)
    if snapshot is not None:
        snapshot.assert_match(rendered, "decompiled.py")

    restored = _rebuild(rendered)

    assert eq(expr.unbind(), restored)


def _rebuild(code: str) -> ir.Expr:
    """Run decompiled Python code and return the expression it builds."""
    namespace = {}
    exec(code, {}, namespace)  # noqa: S102
    return namespace["result"]


def decompile_roundtrip(expr: ir.Expr) -> ir.Expr:
    """Decompile ``expr`` to Python code, run it and return what it builds."""
    return _rebuild(ibis.decompile(expr))


def expose_sort_keys(sql: str, dialect: str | None = None) -> tuple[str, list[int]]:
    """Make the keys of a top-level ``ORDER BY`` output columns of ``sql``.

    Returns the SQL to run and the positions of the keys in its result, which
    are negative for keys appended after the query's own columns. Without a
    top-level ``ORDER BY`` the positions are empty.
    """
    import sqlglot as sg
    import sqlglot.expressions as sge

    query = sg.parse_one(sql, read=dialect)
    if (order := query.args.get("order")) is None:
        return sql, []
    keys = []
    for ordered in order.expressions:
        key = ordered.this
        while isinstance(key, sge.Paren):
            key = key.this
        keys.append(key)

    if not isinstance(query, sge.Select):
        # a set operation orders by an output column or position
        select = query
        while isinstance(select, (sge.Union, sge.Intersect, sge.Except)):
            select = select.left
        names = [item.alias_or_name for item in select.expressions]
        return sql, [
            int(key.this) - 1
            if isinstance(key, sge.Literal) and key.is_int
            else names.index(key.name)
            for key in keys
        ]

    outputs = {
        item.alias_or_name: item.this if isinstance(item, sge.Alias) else item
        for item in query.expressions
        if not item.is_star
    }
    star = any(item.is_star for item in query.expressions)
    positions, appended = [], []
    for key in keys:
        if isinstance(key, sge.Literal) and key.is_int:
            if star:
                # only the result knows which column the star expands this to
                positions.append(int(key.this) - 1)
                continue
            item = query.expressions[int(key.this) - 1]
            key = item.this if isinstance(item, sge.Alias) else item
        elif isinstance(key, sge.Column) and not key.table and key.name in outputs:
            key = outputs[key.name]
        positions.append(None)
        appended.append(key.copy().as_(f"_sort_key_{len(appended)}"))
    if not appended:
        return sql, positions
    from_end = iter(range(-len(appended), 0))
    positions = [next(from_end) if p is None else p for p in positions]
    return query.select(*appended).sql(dialect=dialect), positions


def assert_same_result(
    result: pd.DataFrame,
    expected: pd.DataFrame,
    *,
    sort_keys: Sequence[int] = (),
    narrow_decimals: bool = False,
) -> None:
    """Assert that ``result`` holds the rows an engine returned as ``expected``.

    ``expected`` carries the keys of the query's top-level ``ORDER BY`` at the
    ``sort_keys`` positions (see `expose_sort_keys`); the rows are then
    compared in that order, rows that tie on the keys in any order. Without
    keys the rows are compared as a multiset.

    Column names must match, ignoring case, wherever the SQL names the column;
    an unaliased expression is named by the engine (``count_star()`` in
    DuckDB, ``_col_N`` in Ibis) and is skipped. A column must hold the same
    kind of value on both sides (numbers, strings, booleans, dates and
    times), and the values must be equal, floating-point numbers to one part
    in a billion.

    ``narrow_decimals`` admits one known deviation: a decimal column that Ibis
    types at a scale narrower than the engine computes (``AVG`` of a decimal,
    a product of two decimals) is rounded to that scale when it is fetched,
    so it may differ by half a unit of its last place. Like a strict xfail,
    the flag requires at least one such column to differ.
    """
    import pandas as pd

    width = len(result.columns)
    keys = expected.iloc[:, list(sort_keys)]
    expected = expected.iloc[:, :width]
    assert result.shape == expected.shape, (
        f"the result has shape {result.shape}, the expected rows {expected.shape}"
    )
    mismatched = [
        (name, want)
        for name, want in zip(result.columns, expected.columns)
        if not re.fullmatch(r"_col_\d+", name) and name.lower() != want.lower()
    ]
    assert not mismatched, f"column names differ (result, expected): {mismatched}"

    left, right, rounding = {}, {}, {}
    for i, (name, want) in enumerate(zip(result.columns, expected.columns)):
        kind, expected_kind = _kind(result[name]), _kind(expected[want])
        assert kind == expected_kind or "null" in (kind, expected_kind), (
            f"column {name!r} holds {kind} values in the result "
            f"and {expected_kind} values in the expected rows"
        )
        kind = expected_kind if kind == "null" else kind
        left[i] = _normalize(result[name], kind).to_numpy()
        right[i] = _normalize(expected[want], kind).to_numpy()
        if (scale := _narrower_decimal_scale(result[name], expected[want])) is not None:
            rounding[i] = 0.5 * 10**-scale
    left, right = pd.DataFrame(left), pd.DataFrame(right)

    # sort both sides the same way; a rounded column decides last, so that
    # the rounding cannot pair the rows up differently
    order = [i for i in range(width) if i not in rounding] + list(rounding)
    if sort_keys:
        previous = keys.shift()
        changed = keys.ne(previous) & ~(keys.isna() & previous.isna())
        group = changed.any(axis=1).cumsum().to_numpy()
        left["group"], right["group"] = group, group
        order = ["group", *order]
    left = left.sort_values(order, kind="stable").reset_index(drop=True)
    right = right.sort_values(order, kind="stable").reset_index(drop=True)

    rounded = []
    for i, name in enumerate(result.columns):
        if i in rounding and (row := _first_difference(left[i], right[i])) is not None:
            rounded.append(name)
            assert narrow_decimals, (
                f"column {name!r} differs at row {row}: "
                f"{left[i][row]!r} != {right[i][row]!r}; Ibis types the column at "
                "a narrower decimal scale than the engine computes and rounds it "
                "on fetch, which `narrow_decimals` admits"
            )
        atol = rounding.get(i, 0.0) if narrow_decimals else 0.0
        if (row := _first_difference(left[i], right[i], atol=atol)) is not None:
            raise AssertionError(
                f"column {name!r} differs at row {row}: "
                f"{left[i][row]!r} != {right[i][row]!r}"
            )
    assert not narrow_decimals or rounded, (
        "no decimal column is rounded any more; drop `narrow_decimals`"
    )


def _kind(column: pd.Series) -> str:
    """What a column holds: numbers, strings, booleans, times, or nothing."""
    import numpy as np
    import pandas.api.types as pdt

    values = column.dropna()
    if values.empty:
        return "null"
    if pdt.is_bool_dtype(column):
        return "boolean"
    if pdt.is_datetime64_any_dtype(column):
        return "temporal"
    if pdt.is_numeric_dtype(column):
        return "numeric"
    sample = values.iloc[0]
    if isinstance(sample, (bool, np.bool_)):
        return "boolean"
    if isinstance(sample, numbers.Number):
        return "numeric"
    if isinstance(sample, datetime.date):
        return "temporal"
    if isinstance(sample, str):
        return "string"
    return type(sample).__name__


def _normalize(column: pd.Series, kind: str) -> pd.Series:
    import pandas as pd

    if kind == "numeric":
        return pd.to_numeric(column).astype(float)
    if kind == "temporal":
        values = pd.to_datetime(column)
        return values if values.dt.tz is not None else values.astype("datetime64[ns]")
    return column.astype(object).where(column.notna(), None)


def _decimal_scale(column: pd.Series) -> int | None:
    """The scale of a column of decimals; ``None`` for any other column."""
    values = column.dropna()
    if values.empty or not isinstance(values.iloc[0], decimal.Decimal):
        return None
    return max(0, *(-value.as_tuple().exponent for value in values))


def _narrower_decimal_scale(result: pd.Series, expected: pd.Series) -> int | None:
    """The scale of ``result`` when it holds decimals narrower than ``expected``."""
    import pandas.api.types as pdt

    if (scale := _decimal_scale(result)) is None:
        return None
    wider = _decimal_scale(expected)
    if wider is None and pdt.is_float_dtype(expected):
        return scale
    return scale if wider is not None and wider > scale else None


def _first_difference(
    left: pd.Series, right: pd.Series, *, atol: float = 0.0, rtol: float = 1e-9
) -> int | None:
    """The first row where the two columns differ, or ``None``."""
    import numpy as np

    same = left.isna() & right.isna()
    if left.dtype.kind == "f":
        same |= (left - right).abs() <= atol + 1e-9 + rtol * right.abs()
    else:
        same |= left == right
    if same.all():
        return None
    return int(np.flatnonzero(~same.to_numpy())[0])
