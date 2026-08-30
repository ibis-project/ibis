from __future__ import annotations

import contextlib

import pytest

import ibis
from ibis.common import exceptions as exc
from ibis.expr import datatypes as dt
from ibis.tstring import t

five = ibis.literal(5)
world = ibis.literal("world")


@contextlib.contextmanager
def set_default_backend(backend: str):
    """Context manager to set the default backend temporarily.

    eg
    with set_default_backend('duckdb'):
        ...
    """
    original = ibis.get_backend()
    ibis.set_backend(backend)
    try:
        yield
    finally:
        ibis.set_backend(original)


@pytest.mark.notimpl(["polars"])
@pytest.mark.parametrize(
    ("template", "expected_result"),
    [
        (t("{five} + 3"), 8),
        (t("{five:.2f} + 3"), 8),  # format strings ignored
        (t("'hello ' || {world}"), "hello world"),
        (t("'hello ' || {world!r}"), "hello world"),  # conversion strings ignored
    ],
)
def test_scalar(con, template, expected_result):
    """Test that scalar template expressions execute correctly."""
    expr = ibis.sql_value(template)
    result = con.execute(expr)
    assert result == expected_result


@pytest.mark.parametrize(
    "typ",
    [None, "timestamp('America/Anchorage')"],
)
def test_uninferrable_dtype(typ):
    """Test behavior when a template's dtype can't be inferred using sqlglot"""
    # parse a UTC timestamp into alaska local time, eg "8/1/2024 21:44:00" into 2024-08-01 13:44:00 (8 hours before UTC).
    con = ibis.duckdb.connect()
    timestamp = ibis.timestamp("2024-08-01 21:44:00")  # noqa: F841
    template = t("{timestamp} AT TIME ZONE 'UTC' AT TIME ZONE 'America/Anchorage'")
    val = ibis.sql_value(template, type=typ)
    if typ is None:
        assert val.type().is_unknown()
    else:
        assert val.type() == ibis.dtype(typ)

    # Still, even if the type couldn't be inferred, we can still cast it to string later
    # and everything works.
    in_ak_string = val.cast(str).name("in_ak_time")
    assert isinstance(in_ak_string, ibis.ir.StringScalar)

    expected_sql = '''SELECT
  CAST(MAKE_TIMESTAMP(2024, 8, 1, 21, 44, 0.0) AT TIME ZONE 'UTC' AT TIME ZONE 'America/Anchorage' AS TEXT) AS "in_ak_time"'''
    actual_sql = in_ak_string.to_sql()
    assert actual_sql == expected_sql

    result = con.execute(in_ak_string)
    expected = "2024-08-01 13:44:00"
    assert result == expected


@pytest.mark.notimpl(["polars"])
def test_column(con, alltypes, backend):
    """Test template with column interpolation."""
    c = alltypes.int_col  # noqa: F841
    template = t("{c + 2} - 1")
    expr = ibis.sql_value(template)
    assert isinstance(expr, ibis.ir.IntegerColumn)
    assert expr.type() == dt.int64
    result = con.execute(expr)
    expected = con.execute(alltypes.int_col + 1)
    backend.assert_series_equal(result, expected, check_names=False)


@pytest.mark.notimpl(["polars"])
def test_deferred(con, alltypes, backend):
    """Test template with column interpolation."""
    i = ibis._.int_col  # noqa: F841
    template = t("{i + 2} - 1")
    expr = ibis.sql_value(template)
    assert isinstance(expr, ibis.Deferred)
    with pytest.raises(TypeError):
        # We can't execute a Deferred directly, we need to bind it to the table first
        con.execute(expr)
    (bound,) = alltypes.bind(expr)
    result = con.execute(bound)
    expected = con.execute(alltypes.int_col + 1)
    backend.assert_series_equal(result, expected, check_names=False)


@pytest.mark.notimpl(["polars"])
def test_direct_select(con, alltypes, backend):
    """Test template with column interpolation."""
    i = ibis._.int_col  # noqa: F841
    five = 5  # noqa: F841
    selected = alltypes.select(
        scalar=t("{five} - 1"),
        col=t("{alltypes.int_col + 2} - 1"),
        deferred=t("cast({i + 2} as varchar)"),
    )
    expected = alltypes.select(
        scalar=ibis.literal(4).cast("int32"),
        col=(alltypes.int_col + 1).cast("int64"),
        deferred=(alltypes.int_col + 2).cast("string"),
    )
    actual_schema = selected.schema()
    expected_schema = expected.schema()
    assert expected_schema == actual_schema
    result = con.execute(selected)
    expected_result = con.execute(expected)
    backend.assert_frame_equal(result, expected_result)


def test_sqlite_template_correctly_executed_on_duckdb():
    pa = pytest.importorskip("pyarrow")
    five = ibis.literal(5)  # noqa: F841
    template = t("CAST({five} AS REAL)")

    expr_sqlite = ibis.sql_value(template, dialect="sqlite")
    expr_default = ibis.sql_value(template)

    con_sqlite = ibis.sqlite.connect()
    result = con_sqlite.to_pyarrow(expr_default)
    assert result.type == pa.float32()
    assert result.as_py() == 5.0
    result = con_sqlite.to_pyarrow(expr_sqlite)
    assert result.type == pa.float64()
    assert result.as_py() == 5.0

    con_duckdb = ibis.duckdb.connect()
    result = con_duckdb.to_pyarrow(expr_default)
    assert result.type == pa.float32()
    assert result.as_py() == 5.0
    result = con_duckdb.to_pyarrow(expr_sqlite)
    assert result.type == pa.float64()
    assert result.as_py() == 5.0


@pytest.mark.parametrize(
    "template_dialect,dialect_override,default_backend,expected_dialect",
    [
        # If the template doesn't rely on a backend...
        (None, None, "sqlite", "sqlite"),
        (None, None, "duckdb", "duckdb"),
        (None, "sqlite", "duckdb", "sqlite"),
        (None, "sqlite", "sqlite", "sqlite"),
        (None, "duckdb", "duckdb", "duckdb"),
        (None, "duckdb", "sqlite", "duckdb"),
        # If the template relies on a backend...
        ("sqlite", None, "sqlite", "sqlite"),
        ("sqlite", None, "duckdb", "sqlite"),
        ("sqlite", "sqlite", "duckdb", "sqlite"),
        ("sqlite", "sqlite", "sqlite", "sqlite"),
        ("sqlite", "duckdb", "duckdb", "duckdb"),
        ("sqlite", "duckdb", "sqlite", "duckdb"),
    ],
)
def test_dialect_inferrence(
    template_dialect, dialect_override, default_backend, expected_dialect
):
    if template_dialect is None:
        templ = ibis.t("4 + 5")
    elif template_dialect == "sqlite":
        con = ibis.sqlite.connect()
        table = con.create_table("t1", {"i": [1, 2, 3]})  # noqa: F841
        templ = ibis.t("{table.i} + 5")
    else:
        raise ValueError(f"Unexpected template_dialect: {template_dialect}")

    with set_default_backend(default_backend):
        expr = ibis.sql_value(templ, dialect=dialect_override)
    actual = expr.op().dialect  # ty:ignore[possibly-missing-attribute]
    assert actual == expected_dialect


def test_multiple_backends_errors():
    """If you try to create a sql_value that relies on multiple backends, raise."""
    sqlite1 = ibis.sqlite.connect()
    sqlite2 = ibis.sqlite.connect()
    t1 = sqlite1.create_table("t1", {"i": [1, 2, 3]})
    t2 = sqlite2.create_table("t2", {"i": [4, 5, 6]})
    scalar1 = t1.i.sum()  # noqa: F841
    scalar2 = t2.i.sum()  # noqa: F841

    template_same_backend = ibis.t("{scalar1} + {scalar1}")
    actual = ibis.sql_value(template_same_backend).execute()
    expected = sqlite1.execute(t1.i.sum() + t1.i.sum())
    assert actual == expected

    template_different_backends = ibis.t("{scalar1} + {scalar2}")
    with pytest.raises(
        exc.IbisInputError,
        match="A SQL value can only depend on a single relation, got 2",
    ):
        ibis.sql_value(template_different_backends)


def test_multiple_relations_errors():
    """If you try to create a sql_value that relies on multiple relations, raise."""
    con = ibis.sqlite.connect()
    t1 = con.create_table("t1", {"i": [1, 2, 3]})
    t2 = t1.mutate(i2=t1.i + 10)

    template_same_relation = ibis.t("{t2.i.sum()} + {t2.i2}")
    actual = ibis.sql_value(template_same_relation).execute()
    expected = con.execute(t2.i.sum() + t2.i2)
    assert (actual == expected).all()

    template_different_relations = ibis.t("{t1.i.sum()} + {t2.i2}")
    with pytest.raises(
        exc.IbisInputError,
        match="A SQL value can only depend on a single relation, got 2",
    ):
        ibis.sql_value(template_different_relations)
