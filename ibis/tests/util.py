"""Test utilities."""

from __future__ import annotations

import datetime
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
    import pyarrow as pa


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
    branches = list(_branches(query))

    if [key.name.upper() for key in keys if isinstance(key, sge.Var)] == ["ALL"]:
        # every output column, left to right
        for select in branches:
            if not any(item.is_star for item in select.expressions):
                return sql, list(range(len(select.expressions)))
        # only the result knows what the star expands to; compare as a multiset
        return sql, []

    if not isinstance(query, sge.Select):
        # a set operation orders by an output column, named in any branch
        positions, appended = [], []
        for key in keys:
            if isinstance(key, sge.Literal) and key.is_int:
                positions.append(int(key.this) - 1)
                continue
            position = next(
                (
                    names.index(key.name)
                    for names in (
                        [item.alias_or_name for item in select.expressions]
                        for select in branches
                        if not any(item.is_star for item in select.expressions)
                    )
                    if key.name in names
                ),
                None,
            )
            if position is None:
                appended.append(key.copy().as_(f"_sort_key_{len(appended)}"))
            positions.append(position)
        if not appended:
            return sql, positions
        # a star hides the name: order a wrapping query that exposes the key
        inner = query.copy()
        for arg in ("order", "limit", "offset"):
            inner.set(arg, None)
        wrapped = sg.select(sge.Star(), *appended).from_(inner.subquery("_q"))
        for arg in ("order", "limit", "offset"):
            wrapped.set(arg, query.args.get(arg))
        from_end = iter(range(-len(appended), 0))
        positions = [next(from_end) if p is None else p for p in positions]
        return wrapped.sql(dialect=dialect), positions

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
        elif query.args.get("distinct") and key not in outputs.values():
            # projecting the key would widen the DISTINCT set; keep the keys
            # before it, whose ties it only breaks
            break
        positions.append(None)
        appended.append(key.copy().as_(f"_sort_key_{len(appended)}"))
    if not appended:
        return sql, positions
    from_end = iter(range(-len(appended), 0))
    positions = [next(from_end) if p is None else p for p in positions]
    return query.select(*appended).sql(dialect=dialect), positions


def unnamed_columns(sql: str, dialect: str | None = None) -> list[int]:
    """The positions of the output columns ``sql`` leaves the engine to name.

    A position after a star counts from the end of the query's own columns.
    """
    import sqlglot as sg
    import sqlglot.expressions as sge

    select = next(_branches(sg.parse_one(sql, read=dialect)))
    items = select.expressions
    positions, star = [], False
    for i, item in enumerate(items):
        if item.is_star:
            star = True
        elif not isinstance(item, (sge.Alias, sge.Column)):
            positions.append(i - len(items) if star else i)
    return positions


def _branches(query):
    """The ``SELECT`` statements whose rows a query returns, leftmost first."""
    import sqlglot.expressions as sge

    while isinstance(query, sge.Subquery):
        query = query.this
    if isinstance(query, sge.SetOperation):
        yield from _branches(query.left)
        yield from _branches(query.right)
    else:
        yield query


def assert_same_result(
    result: pa.Table | pd.DataFrame,
    expected: pa.Table | pd.DataFrame,
    *,
    sort_keys: Sequence[int] = (),
    unnamed: Sequence[int] = (),
) -> None:
    """Assert that ``result`` holds the rows an engine returned as ``expected``.

    ``expected`` carries the keys of the query's top-level ``ORDER BY`` at the
    ``sort_keys`` positions (see `expose_sort_keys`); the rows are then
    compared in that order, rows that tie on the keys in any order. Without
    keys the rows are compared as a multiset.

    Column names must match, ignoring case, except at the ``unnamed``
    positions, where the SQL leaves the engine to name the column (see
    `unnamed_columns`); a name the SQL repeats may carry a suffix (``a_1``). Given Arrow tables, the engine must type each column
    the same way on both sides, but for the width of an integer and the
    precision of a decimal. A column must hold the same kind of value on both
    sides (numbers, strings, booleans, dates and times), and the values must
    be equal, floating-point numbers to one part in a billion.
    """
    import pandas as pd
    import pyarrow as pa

    width = len(expected.columns) - sum(position < 0 for position in sort_keys)
    assert len(result.columns) == width, (
        f"the result has {len(result.columns)} columns, the query {width}"
    )
    if isinstance(expected, pa.Table):
        mistyped = [
            (field.name, field.type, want.type)
            for field, want in zip(result.schema, expected.schema)
            if _engine_type(field.type) != _engine_type(want.type)
        ]
        assert not mistyped, f"column types differ (name, result, expected): {mistyped}"
        result = result.to_pandas(integer_object_nulls=True)
        expected = expected.to_pandas(integer_object_nulls=True)

    keys = expected.iloc[:, list(sort_keys)]
    expected = expected.iloc[:, :width]
    assert result.shape == expected.shape, (
        f"the result has shape {result.shape}, the expected rows {expected.shape}"
    )
    unnamed = {position % width for position in unnamed}
    mismatched, seen = [], set()
    for i, (name, want) in enumerate(zip(result.columns, expected.columns)):
        # a repeated name may be told apart with a suffix, as pandas needs
        repeated = want.lower() in seen
        seen.add(want.lower())
        if i in unnamed or name.lower() == want.lower():
            continue
        if not (
            repeated and re.fullmatch(rf"{re.escape(want)}_\d+", name, re.IGNORECASE)
        ):
            mismatched.append((name, want))
    assert not mismatched, f"column names differ (result, expected): {mismatched}"

    left, right = {}, {}
    for i, name in enumerate(result.columns):
        got, want = result.iloc[:, i], expected.iloc[:, i]
        kind, expected_kind = _kind(got), _kind(want)
        assert kind == expected_kind or "null" in (kind, expected_kind), (
            f"column {name!r} holds {kind} values in the result "
            f"and {expected_kind} values in the expected rows"
        )
        kind = expected_kind if kind == "null" else kind
        if kind == "numeric" and not (_holds_floats(got) or _holds_floats(want)):
            # integers and decimals compare exactly
            kind = "exact"
        left[i] = _normalize(got, kind).to_numpy()
        right[i] = _normalize(want, kind).to_numpy()
    left, right = pd.DataFrame(left), pd.DataFrame(right)

    # sort both sides the same way
    order = list(range(width))
    if sort_keys:
        previous = keys.shift()
        changed = keys.ne(previous) & ~(keys.isna() & previous.isna())
        group = changed.any(axis=1).cumsum().to_numpy()
        left["group"], right["group"] = group, group
        order = ["group", *order]
    left = left.sort_values(order, kind="stable").reset_index(drop=True)
    right = right.sort_values(order, kind="stable").reset_index(drop=True)

    for i, name in enumerate(result.columns):
        if (row := _first_difference(left[i], right[i])) is not None:
            raise AssertionError(
                f"column {name!r} differs at row {row}: "
                f"{left[i][row]!r} != {right[i][row]!r}"
            )


def _engine_type(dtype: pa.DataType) -> object:
    """A column type, ignoring the width of an integer and a decimal's precision."""
    import pyarrow as pa

    if pa.types.is_signed_integer(dtype):
        return "signed integer"
    if pa.types.is_unsigned_integer(dtype):
        return "unsigned integer"
    if pa.types.is_decimal(dtype):
        return f"decimal with scale {dtype.scale}"
    return dtype


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


def _holds_floats(column: pd.Series) -> bool:
    if column.dtype.kind == "f":
        return True
    return column.dtype == object and any(
        isinstance(value, float) for value in column.dropna()
    )


def _normalize(column: pd.Series, kind: str) -> pd.Series:
    import pandas as pd

    if kind == "numeric":
        return pd.to_numeric(column).astype(float)
    if kind == "temporal":
        if column.dtype == object:
            # dates; microseconds reach past the year 2262, unlike nanoseconds
            column = column.astype("datetime64[us]")
        values = pd.to_datetime(column)
        return values if values.dt.tz is not None else values.astype("datetime64[us]")
    if kind not in ("string", "boolean", "exact", "null"):
        # an interval, list or struct does not sort; its text does
        column = column.map(_text, na_action="ignore")
    return column.astype(object).where(column.notna(), None)


def _text(value: object) -> str:
    import numpy as np

    return repr(value.tolist() if isinstance(value, np.ndarray) else value)


def _first_difference(
    left: pd.Series, right: pd.Series, *, rtol: float = 1e-9
) -> int | None:
    """The first row where the two columns differ, or ``None``."""
    import numpy as np

    same = (left.isna() & right.isna()) | (left == right)
    if left.dtype.kind == "f":
        same |= (left - right).abs() <= 1e-9 + rtol * right.abs()
    if same.all():
        return None
    return int(np.flatnonzero(~same.to_numpy())[0])
