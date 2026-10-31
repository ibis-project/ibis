from __future__ import annotations

import datetime
import decimal
import string
import uuid

import pytest
import sqlglot as sg

import ibis
import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
from ibis.backends.db2.tests.conftest import (
    DB2_DATABASE,
    DB2_HOST,
    DB2_PASS,
    DB2_PORT,
    DB2_USER,
)
from ibis.util import gen_name


def test_connect_url():
    new_con = ibis.connect(
        f"db2://{DB2_USER}:{DB2_PASS}@{DB2_HOST}:{DB2_PORT}/{DB2_DATABASE}"
    )
    result = new_con.raw_sql("SELECT 1 FROM SYSIBM.SYSDUMMY1").fetchone()[0]
    assert result == 1


def test_list_tables(con):
    tables = con.list_tables()
    assert set(tables) >= {
        "astronauts",
        "awards_players",
        "batting",
        "diamonds",
        "functional_alltypes",
        "win",
    }


def test_current_database(con):
    db = con.current_database
    assert isinstance(db, str)
    assert len(db) > 0


# Db2 blank-pads the CURRENT SCHEMA register to 8 characters.
@pytest.mark.parametrize("length", [7, 8, 20])
def test_current_database_is_not_padded(con, length):
    schema = f"S{uuid.uuid4().hex.upper()}"[:length]
    table = gen_name("current_database")

    with con._safe_raw_sql(f'CREATE SCHEMA "{schema}"'):
        pass
    try:
        with con._safe_raw_sql(f'CREATE TABLE "{schema}"."{table}" (x INTEGER)'):
            pass
        try:
            other = ibis.db2.connect(
                host=DB2_HOST,
                port=DB2_PORT,
                database=DB2_DATABASE,
                user=DB2_USER,
                password=DB2_PASS,
                schema=schema,
            )
            try:
                assert other.current_database == schema
                assert other.current_database in other.list_databases()
                # Db2 compares schema names without trailing blanks, so this
                # passes even with a padded name; it guards the lookup itself.
                assert other.list_tables(database=other.current_database) == [table]
            finally:
                other.disconnect()
        finally:
            with con._safe_raw_sql(f'DROP TABLE "{schema}"."{table}"'):
                pass
    finally:
        with con._safe_raw_sql(f'DROP SCHEMA "{schema}" RESTRICT'):
            pass


def test_version(con):
    v = con.version
    assert isinstance(v, str)
    assert len(v) > 0


@pytest.mark.parametrize(
    ("server_type", "expected_type"),
    [
        ("SMALLINT", dt.int16),
        ("INTEGER", dt.int32),
        ("BIGINT", dt.int64),
        ("REAL", dt.float32),
        ("DOUBLE", dt.float64),
        ("DATE", dt.date),
        ("TIME", dt.time),
        ("TIMESTAMP", dt.timestamp),
        ("VARCHAR(100)", dt.string),
        ("CHAR(10)", dt.string),
        ("CLOB", dt.string),
    ],
)
def test_get_schema_types(con, server_type, expected_type, temp_table):
    with con._safe_raw_sql(f'CREATE TABLE "{temp_table}" (x {server_type})'):
        pass
    con.con.commit()

    try:
        schema = con.get_schema(temp_table)
        # DB2 uppercases unquoted column names in SYSCAT.COLUMNS ("x" -> "X")
        key = "x" if "x" in schema else "X"
        assert isinstance(schema[key], type(expected_type))
    finally:
        con.drop_table(temp_table, force=True)


def test_get_schema(con):
    schema = con.get_schema("functional_alltypes")
    assert "id" in schema
    assert "bool_col" in schema
    assert "string_col" in schema
    assert "timestamp_col" in schema


def test_create_and_drop_table(con):
    name = gen_name("db2_test")
    t = con.create_table(name, schema={"a": "int32", "b": "string"})
    try:
        assert len(t.execute()) == 0
        assert set(t.columns) == {"a", "b"}
    finally:
        con.drop_table(name, force=True)


def test_drop_nonexistent_table_force(con):
    # should not raise
    con.drop_table(gen_name("db2_nonexistent"), force=True)


def test_insert_and_select(con):
    import pandas as pd

    name = gen_name("db2_test")
    con.create_table(name, schema={"x": "int64", "y": "string"})
    try:
        df = pd.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})
        con.insert(name, df)
        t = con.table(name)
        result = t.order_by("x").execute()
        assert len(result) == 3
        assert list(result["y"]) == ["a", "b", "c"]
    finally:
        con.drop_table(name, force=True)


def test_overwrite_insert(con):
    import pandas as pd

    name = gen_name("db2_test")
    df1 = pd.DataFrame({"x": [1, 2, 3]})
    df2 = pd.DataFrame({"x": [10, 20]})
    con.create_table(name, df1)
    try:
        con.insert(name, df2, overwrite=True)
        t = con.table(name)
        result = t.execute()
        assert len(result) == 2
        assert sorted(result["x"].tolist()) == [10, 20]
    finally:
        con.drop_table(name, force=True)


def test_execute_keeps_decimal_exact(con):
    import pandas as pd

    values = [
        decimal.Decimal("123456789012345678.99"),
        decimal.Decimal("-0.01"),
        None,
    ]
    name = gen_name("db2_test")
    con.create_table(name, schema={"d": "decimal(20, 2)"})
    try:
        con.insert(name, pd.DataFrame({"d": values}))
        result = con.table(name).execute()["d"]
        assert sorted(result.dropna()) == sorted(v for v in values if v is not None)
        assert result.isna().sum() == 1
    finally:
        con.drop_table(name, force=True)


def test_execute_decimal_wider_than_float(con):
    # 25 nines do not fit in a float64; they used to raise TypeError.
    value = decimal.Decimal("9" * 25 + "." + "9" * 6)
    result = con.execute(ibis.literal(value, type="decimal(31, 6)"))
    assert result == value


@pytest.mark.parametrize(
    ("key", "override"),
    [
        ("PWD", {"password": "secret;junk"}),
        ("UID", {"user": "someone;else"}),
        ("HOSTNAME", {"host": "localhost;x"}),
        ("DATABASE", {"database": "db;x"}),
        ("CURRENTSCHEMA", {"schema": "s;x"}),
        ("CONNECTTIMEOUT", {"connecttimeout": "5;x"}),
    ],
)
def test_connect_rejects_semicolon(key, override):
    # Without the check the driver ends the value at `;`, so `secret;junk`
    # would be accepted as `secret`.
    kwargs = {
        "database": DB2_DATABASE,
        "host": DB2_HOST,
        "port": DB2_PORT,
        "user": DB2_USER,
        "password": DB2_PASS,
        **override,
    }
    with pytest.raises(ValueError, match=key):
        ibis.db2.connect(**kwargs)


@pytest.fixture
def connect_dsn(monkeypatch):
    """Record the connection string instead of connecting."""
    import ibm_db_dbi

    seen = []

    def connect(dsn, **_):
        seen.append(dsn)
        raise RuntimeError("no connection attempted")

    monkeypatch.setattr(ibm_db_dbi, "connect", connect)
    return seen


@pytest.mark.parametrize("key", ["hostname", "HostName", "HOSTNAME"])
def test_connect_rejects_hostname(connect_dsn, key):
    # `hostname` used to be silently dropped, so the connection went to
    # `localhost` whatever the caller wrote.
    with pytest.raises(ValueError, match="use `host` instead"):
        ibis.db2.connect(database="d", host="h1", **{key: "h2"})
    assert not connect_dsn


def test_connect_host_and_driver_options_are_passed_on(connect_dsn):
    with pytest.raises(RuntimeError, match="no connection attempted"):
        ibis.db2.connect(database="d", host="h1", connecttimeout=5)
    (dsn,) = connect_dsn
    assert "HOSTNAME=h1" in dsn
    assert "CONNECTTIMEOUT=5" in dsn


@pytest.mark.parametrize("key", ["hostname", "HostName", "HOSTNAME"])
def test_connect_url_rejects_hostname(connect_dsn, key):
    # The URL already names a host (h1), so a `hostname` query parameter must
    # not be dropped silently either.
    with pytest.raises(ValueError, match="use `host` instead"):
        ibis.connect(f"db2://u:p@h1:50000/d?{key}=h2")
    assert not connect_dsn


def test_connect_url_host_and_driver_options_are_passed_on(connect_dsn):
    with pytest.raises(RuntimeError, match="no connection attempted"):
        ibis.connect("db2://u:p@h1:50000/d?connecttimeout=5")
    (dsn,) = connect_dsn
    assert "HOSTNAME=h1" in dsn
    assert "CONNECTTIMEOUT=5" in dsn


@pytest.mark.parametrize(
    ("make_expr", "expected"),
    [
        pytest.param(lambda t: t.x / t.x.sum(), [0.1, 0.2, 0.3, 0.4], id="sum"),
        pytest.param(lambda t: t.x / t.count(), [0.25, 0.5, 0.75, 1.0], id="count"),
        pytest.param(lambda t: t.x / t.y, [0.5] * 4, id="column"),
        pytest.param(lambda t: t.x / 2, [0.5, 1.0, 1.5, 2.0], id="literal"),
        pytest.param(lambda t: t.x / t.x, [1.0] * 4, id="self"),
    ],
)
def test_divide_integers_is_not_integer_division(con, make_expr, expected):
    # Used to return whole numbers (0.0 for `x / x.sum()`): the integer-to-DOUBLE
    # cast that sqlglot adds was lost when `compile` generated SQL without copying.
    t = ibis.memtable({"x": [1, 2, 3, 4], "y": [2, 4, 6, 8]})
    expr = t.mutate(r=make_expr(t)).order_by("x")
    assert con.execute(expr)["r"].tolist() == pytest.approx(expected)


def test_divide_integers_after_filter(con):
    t = ibis.memtable({"x": [1, 2, 3, 4]})
    filtered = t.filter(t.x > 1)
    expr = filtered.mutate(r=filtered.x / filtered.x.sum()).order_by("x")
    assert con.execute(expr)["r"].tolist() == pytest.approx([2 / 9, 3 / 9, 4 / 9])


def test_divide_decimal_and_float_are_unchanged(con):
    t = ibis.memtable(
        {"f": [1.0, 3.0], "d": [decimal.Decimal("1.50"), decimal.Decimal("3.50")]},
        schema={"f": "float64", "d": "decimal(10, 2)"},
    )
    expr = t.mutate(rf=t.f / t.f.sum(), rd=t.d / t.d.sum()).order_by("f")
    result = con.execute(expr)
    assert result["rf"].tolist() == pytest.approx([0.25, 0.75])
    assert [float(v) for v in result["rd"]] == pytest.approx([0.3, 0.7], abs=1e-3)


@pytest.fixture
def arrow_table():
    return ibis.memtable(
        {"k": [1, 2, 3, 4], "i": [1, 2, 3, None], "b": [True, False, None, True]},
        schema={"k": "int32", "i": "int32", "b": "boolean"},
    )


@pytest.mark.parametrize(
    ("make_expr", "expected"),
    [
        pytest.param(lambda t: t.i * 1.5, [1.5, 3.0, 4.5, None], id="times_float"),
        pytest.param(lambda t: t.i + 0.5, [1.5, 2.5, 3.5, None], id="plus_float"),
        pytest.param(lambda t: t.i * 2.0, [2.0, 4.0, 6.0, None], id="times_float_2"),
        pytest.param(lambda t: t.i - 0.25, [0.75, 1.75, 2.75, None], id="minus_float"),
        pytest.param(lambda t: t.i % 1.5, [1.0, 0.5, 0.0, None], id="mod_float"),
        pytest.param(lambda t: t.i > 1, [False, True, True, None], id="comparison"),
        pytest.param(
            lambda t: ibis.ifelse(t.b, 0.5, 1.5), [0.5, 1.5, 1.5, 0.5], id="ifelse"
        ),
        pytest.param(lambda t: t.b, [True, False, None, True], id="boolean_column"),
    ],
)
def test_to_pyarrow_column(con, arrow_table, make_expr, expected):
    # Used to raise ArrowInvalid: the driver returns Decimal for `int * 1.5`
    # and 0/1 for booleans, and pyarrow does not convert either.
    expr = arrow_table.mutate(v=make_expr(arrow_table)).order_by("k").select("v")
    result = con.to_pyarrow(expr)
    assert result.schema.equals(expr.schema().to_pyarrow())
    assert result["v"].to_pylist() == expected


@pytest.mark.parametrize(
    ("make_expr", "expected"),
    [
        pytest.param(lambda t: t.b.any(), True, id="any"),
        pytest.param(lambda t: t.b.all(), False, id="all"),
        pytest.param(lambda t: (t.i > 1).any(), True, id="comparison_any"),
        pytest.param(lambda t: (t.i > 1).all(), False, id="comparison_all"),
    ],
)
def test_to_pyarrow_boolean_scalar(con, arrow_table, make_expr, expected):
    result = con.to_pyarrow(make_expr(arrow_table))
    assert result.type == "bool"
    assert result.as_py() is expected


def test_to_pyarrow_batches_are_normalized_in_every_chunk(con, arrow_table):
    expr = arrow_table.mutate(v=arrow_table.i * 1.5).order_by("k")
    batches = list(con.to_pyarrow_batches(expr, chunk_size=3))
    assert [len(b) for b in batches] == [3, 1]
    values = [v for b in batches for v in b.column("v").to_pylist()]
    assert values == [1.5, 3.0, 4.5, None]


def test_to_pyarrow_empty_result(con, arrow_table):
    expr = arrow_table.filter(arrow_table.k > 100).mutate(v=arrow_table.i * 1.5)
    result = con.to_pyarrow(expr)
    assert result.num_rows == 0
    assert result.schema.equals(expr.schema().to_pyarrow())


def test_to_pyarrow_join_with_computed_columns(con, arrow_table):
    other = arrow_table.view()
    right = other.select("k", flag=other.b)
    joined = arrow_table.join(right, "k")
    expr = joined.select("k", scaled=joined.i * 1.5, flag=joined.flag).order_by("k")
    result = con.to_pyarrow(expr)
    assert result.schema.equals(expr.schema().to_pyarrow())
    assert result["scaled"].to_pylist() == [1.5, 3.0, 4.5, None]
    assert result["flag"].to_pylist() == [True, False, None, True]


def test_cast_float_to_boolean(con):
    # Used to fail with SQL0440N: Db2 cannot CAST a DOUBLE to BOOLEAN.
    t = ibis.memtable({"f": [0.0, 2.5, -1.0, None]}, schema={"f": "float64"})
    result = con.execute(t.f.cast("boolean"))
    assert result.tolist() == [False, True, True, None]


def test_sql_accepts_current_date_special_register(con):
    # Used to parse as `CURRENT AS DATE` or raise ParseError.
    t = con.sql(
        "SELECT CURRENT DATE AS d, CURRENT TIMESTAMP AS ts FROM SYSIBM.SYSDUMMY1"
    )
    assert t.schema() == ibis.schema({"D": "!date", "TS": "!timestamp(6)"})
    assert len(t.execute()) == 1


def test_cast_timestamp_to_string(con):
    # Used to return 2020-01-31-10.11.12.345678, with dashes and dots.
    t = ibis.memtable(
        {
            "ts": [
                datetime.datetime(2020, 1, 31, 10, 11, 12, 345678),
                datetime.datetime(2020, 1, 31, 10, 11, 12),
                None,
            ]
        },
        schema={"ts": "timestamp(6)"},
    )
    result = con.execute(t.ts.cast("string"))
    assert result.tolist() == [
        "2020-01-31 10:11:12.345678",
        "2020-01-31 10:11:12.000000",
        None,
    ]


_RANDOM_T = ibis.table({"x": "int", "y": "int"}, name="t")
_RANDOM_U = ibis.table({"z": "int"}, name="u")
_RANDOM_SUBQUERY = _RANDOM_U.filter(ibis.random() < 0.5).z.count().as_scalar()


@pytest.mark.parametrize(
    "window",
    [
        pytest.param({"order_by": ibis.random()}, id="order_by"),
        pytest.param({"order_by": ibis.random() + 1}, id="order_by_nested"),
        pytest.param(
            {"order_by": [_RANDOM_T.x, ibis.random()]}, id="order_by_second_key"
        ),
        pytest.param(
            {"group_by": ibis.random(), "order_by": _RANDOM_T.y}, id="group_by"
        ),
        pytest.param(
            {"group_by": ibis.random() + 1, "order_by": _RANDOM_T.y},
            id="group_by_nested",
        ),
    ],
)
def test_window_keyed_by_random_is_not_defined(window):
    # Db2 rejects RAND() and RANDOM() in a window's ORDER BY or PARTITION BY (SQL0583N).
    expr = _RANDOM_T.select(c=ibis.ntile(2).over(**window))
    with pytest.raises(com.OperationNotDefinedError, match="by a random value"):
        ibis.to_sql(expr, dialect="db2")


@pytest.mark.parametrize(
    "expr",
    [
        pytest.param(_RANDOM_T.select(r=ibis.random()), id="select"),
        pytest.param(_RANDOM_T.filter(ibis.random() < 0.5), id="filter"),
        pytest.param(_RANDOM_T.order_by(ibis.random()), id="order_by"),
        pytest.param(
            _RANDOM_T.select(
                c=(_RANDOM_T.x + ibis.random()).sum().over(order_by=_RANDOM_T.y)
            ),
            id="window_argument",
        ),
        pytest.param(
            _RANDOM_T.select(
                c=ibis.ntile(2).over(order_by=_RANDOM_SUBQUERY),
            ),
            id="scalar_subquery_in_window_order_by",
        ),
        pytest.param(
            _RANDOM_T.select(
                c=ibis.ntile(2).over(group_by=_RANDOM_SUBQUERY, order_by=_RANDOM_T.y),
            ),
            id="scalar_subquery_in_window_group_by",
        ),
    ],
)
def test_random_outside_a_window_key_still_compiles(expr):
    assert "RANDOM()" in ibis.to_sql(expr, dialect="db2")


@pytest.mark.parametrize("name", ["DAYOFWEEK", "DAYOFYEAR", "DAYOFMONTH"])
def test_day_functions_keep_their_db2_names(name):
    # sqlglot parses these into nodes it prints as DAY_OF_WEEK, DAY_OF_YEAR
    # and DAY_OF_MONTH. Db2 has no such functions (SQL0440N).
    sql = sg.parse_one(f"SELECT {name}(d) FROM t", read="db2").sql(dialect="db2")
    assert f"{name}(d)" in sql
    assert "DAY_OF" not in sql


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        pytest.param("DAYOFWEEK(DATE('2024-01-07'))", 1, id="dayofweek_sunday"),
        pytest.param("DAYOFWEEK(DATE('2024-01-13'))", 7, id="dayofweek_saturday"),
        pytest.param("DAYOFYEAR(DATE('2024-03-01'))", 61, id="dayofyear_leap"),
        pytest.param("DAYOFYEAR(DATE('2024-12-31'))", 366, id="dayofyear_last"),
        pytest.param("DAYOFMONTH(DATE('2024-02-29'))", 29, id="dayofmonth_leap"),
        pytest.param("DAYOFMONTH(DATE('2024-01-01'))", 1, id="dayofmonth_first"),
        pytest.param(
            "DAYOFWEEK(TIMESTAMP('2024-01-07 23:59:59'))", 1, id="dayofweek_timestamp"
        ),
        pytest.param(
            "DAYOFYEAR(TIMESTAMP('2024-03-01 00:00:00'))", 61, id="dayofyear_timestamp"
        ),
        pytest.param(
            "DAYOFMONTH(TIMESTAMP('2024-02-29 12:30:00'))",
            29,
            id="dayofmonth_timestamp",
        ),
        pytest.param(
            "DAYOFWEEK(DATE('2024-01-07')) + DAYOFYEAR(DATE('2024-03-01'))"
            " + DAYOFMONTH(DATE('2024-02-29'))",
            1 + 61 + 29,
            id="nested",
        ),
        pytest.param(
            "MOD(DAYOFYEAR(DATE('2024-12-31')), DAYOFMONTH(DATE('2024-01-07')))",
            366 % 7,
            id="nested_in_function",
        ),
    ],
)
def test_sql_accepts_day_functions(con, expr, expected):
    t = con.sql(f'SELECT {expr} AS "v" FROM SYSIBM.SYSDUMMY1')
    assert t.execute()["v"].tolist() == [expected]


@pytest.mark.parametrize("name", ["DAYOFWEEK", "DAYOFYEAR", "DAYOFMONTH"])
@pytest.mark.parametrize("arg_type", ["DATE", "TIMESTAMP"])
def test_sql_day_functions_propagate_null(con, name, arg_type):
    t = con.sql(f'SELECT {name}(CAST(NULL AS {arg_type})) AS "v" FROM SYSIBM.SYSDUMMY1')
    assert t.execute()["v"].isna().all()


def test_sql_day_functions_over_a_column(con, temp_table):
    # The ibis expression API compiles to the same Db2 functions, so both
    # routes must agree row by row.
    with con._safe_raw_sql(f'CREATE TABLE "{temp_table}" ("ts" TIMESTAMP)'):
        pass
    try:
        t = con.table(temp_table)
        con.insert(
            temp_table,
            ibis.memtable(
                {
                    "ts": [
                        datetime.datetime(2024, 1, 7),
                        datetime.datetime(2024, 3, 1, 10, 30),
                        datetime.datetime(2024, 12, 31, 23, 59, 59),
                        None,
                    ]
                },
                schema={"ts": "timestamp(6)"},
            ),
        )
        query = con.sql(
            'SELECT DAYOFWEEK("ts") AS "w", DAYOFYEAR("ts") AS "y",'
            f' DAYOFMONTH("ts") AS "m" FROM "{temp_table}" ORDER BY "ts" NULLS LAST'
        )
        result = con.execute(query)
        assert result["w"].tolist()[:3] == [1, 6, 3]
        assert result["y"].tolist()[:3] == [7, 61, 366]
        assert result["m"].tolist()[:3] == [7, 1, 31]
        assert result[["w", "y", "m"]].iloc[3].isna().all()

        ordered = t.order_by(ibis.asc("ts", nulls_first=False))
        assert con.execute(ordered.ts.day_of_year()).tolist()[:3] == [7, 61, 366]
    finally:
        con.drop_table(temp_table, force=True)


@pytest.mark.parametrize(
    ("query", "modifier"),
    [
        pytest.param("CAST(a AS INTEGER FORMAT 'X')", "format", id="format"),
        pytest.param(
            "CAST(a AS INTEGER DEFAULT 0 ON CONVERSION ERROR)", "default", id="default"
        ),
    ],
)
def test_cast_modifiers_are_rejected(query, modifier):
    tree = sg.parse_one(f"SELECT {query} FROM t", read="db2")
    with pytest.raises(sg.errors.UnsupportedError, match=f"CAST {modifier}"):
        tree.sql(dialect="db2", unsupported_level=sg.ErrorLevel.RAISE)


def test_cast_modifiers_are_rejected_by_the_backend(con):
    t = con.sql(
        "SELECT CAST(1 AS INTEGER DEFAULT 0 ON CONVERSION ERROR) AS n"
        " FROM SYSIBM.SYSDUMMY1",
        schema={"n": "int32"},
    )
    with pytest.raises(com.UnsupportedOperationError, match="CAST default"):
        con.compile(t)


def test_cast_action_is_rejected():
    cast = sg.parse_one("CAST(a AS INTEGER)", read="db2")
    cast.set("action", sg.exp.var("ACTION"))
    with pytest.raises(sg.errors.UnsupportedError, match="CAST action"):
        cast.sql(dialect="db2", unsupported_level=sg.ErrorLevel.RAISE)


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        pytest.param(
            lambda t: t.s.cast("int32"), 'CAST("t0"."s" AS INTEGER)', id="to_int"
        ),
        pytest.param(
            lambda t: t.s.try_cast("int32"), 'CAST("t0"."s" AS INTEGER)', id="try_cast"
        ),
        pytest.param(
            lambda t: t.s.cast("decimal(10, 2)"),
            'CAST("t0"."s" AS DECIMAL(10, 2))',
            id="to_decimal",
        ),
    ],
)
def test_ibis_casts_are_unchanged(expr, expected):
    sql = ibis.to_sql(expr(ibis.table({"s": "string"}, name="t")), dialect="db2")
    assert expected in sql


def test_cast_of_integer_division_keeps_the_cast():
    # Postgres turns CAST(DIV(a, b) AS DECIMAL) into a bare DIV(a, b).
    # Db2 has no DIV function, so keep the CAST as written.
    cast = sg.exp.Cast(
        this=sg.exp.IntDiv(this=sg.column("a"), expression=sg.column("b")),
        to=sg.exp.DataType.build("decimal"),
    )
    assert cast.sql(dialect="db2") == "CAST(DIV(a, b) AS DECIMAL)"
    assert cast.sql(dialect="postgres") == "DIV(a, b)"


@pytest.mark.parametrize(
    ("make_expr", "expected"),
    [
        # `default=` alone used to compile to LAG(a, 0): the default became the
        # offset, so the column came back unshifted.
        pytest.param(lambda c: c.lag(default=0), [0, 10, 20, 30, 40], id="lag_default"),
        pytest.param(
            lambda c: c.lead(default=0), [20, 30, 40, 50, 0], id="lead_default"
        ),
        pytest.param(
            lambda c: c.lag(offset=2, default=0),
            [0, 0, 10, 20, 30],
            id="lag_offset_default",
        ),
        pytest.param(
            lambda c: c.lead(offset=2, default=0),
            [30, 40, 50, 0, 0],
            id="lead_offset_default",
        ),
    ],
)
def test_lag_lead_default(con, make_expr, expected):
    t = ibis.memtable(
        {"k": [1, 2, 3, 4, 5], "a": [10, 20, 30, 40, 50]},
        schema={"k": "int64", "a": "int64"},
    )
    expr = t.select("k", v=make_expr(t.a).over(ibis.window(order_by=t.k)))
    result = con.to_pyarrow(expr.order_by("k"))
    assert result["v"].to_pylist() == expected


def test_strip_family_whitespace(con):
    # Shared tests only strip digits; the whitespace test is not run on Db2.
    values = ["  a b  ", "\t\n x \r\n", "   ", "", None, "ab", "\x0b\x0cz\x0b"]
    t = ibis.memtable(
        {"k": list(range(len(values))), "s": values},
        schema={"k": "int64", "s": "string"},
    )
    for method in ("strip", "lstrip", "rstrip"):
        expr = t.select("k", v=getattr(t.s, method)()).order_by("k")
        expected = [
            None if v is None else getattr(v, method)(string.whitespace) for v in values
        ]
        assert con.to_pyarrow(expr)["v"].to_pylist() == expected, method


@pytest.mark.parametrize(
    ("sep", "expected"),
    [
        pytest.param(",", ["x,y,z", "q", "", "p,r"], id="skips_nulls"),
        pytest.param(", ", ["x, y, z", "q", "", "p, r"], id="multi_char_sep"),
        pytest.param(None, [None] * 4, id="null_sep"),
    ],
)
def test_string_join_null_semantics(con, sep, expected):
    # Db2 has no CONCAT_WS; the compiler reproduces its NULL handling.
    t = ibis.memtable(
        {
            "k": [1, 2, 3, 4],
            "a": ["x", None, None, "p"],
            "b": ["y", "q", None, None],
            "c": ["z", None, None, "r"],
        },
        schema={"k": "int64", "a": "string", "b": "string", "c": "string"},
    )
    sep = ibis.literal(sep, type="string")
    expr = t.select("k", v=sep.join([t.a, t.b, t.c])).order_by("k")
    assert con.to_pyarrow(expr)["v"].to_pylist() == expected


def test_number_cast_to_timestamp(con):
    # Db2 cannot CAST a number to TIMESTAMP. Shared tests cover only 0..9.
    epoch = datetime.datetime(1970, 1, 1)

    def at(seconds, microseconds=0):
        return epoch + datetime.timedelta(seconds=seconds, microseconds=microseconds)

    ints = ibis.memtable(
        {"k": [1, 2, 3, 4, 5], "v": [0, 86400, -1, 1700000000, None]},
        schema={"k": "int64", "v": "int64"},
    )
    result = con.to_pyarrow(ints.select("k", v=ints.v.cast("timestamp")).order_by("k"))
    assert result["v"].to_pylist() == [at(0), at(86400), at(-1), at(1700000000), None]

    floats = ibis.memtable(
        {"k": [1, 2, 3, 4], "v": [1.5, -0.5, 0.0, None]},
        schema={"k": "int64", "v": "float64"},
    )
    result = con.to_pyarrow(
        floats.select("k", v=floats.v.cast("timestamp")).order_by("k")
    )
    assert result["v"].to_pylist() == [at(1, 500000), at(-1, 500000), at(0), None]


@pytest.mark.parametrize("hours", [5.5, -8], ids=["east", "west"])
def test_timestamp_literal_with_offset_raises(con, hours):
    tz = datetime.timezone(datetime.timedelta(hours=hours))
    expr = ibis.literal(datetime.datetime(2020, 1, 2, 3, 4, 5, tzinfo=tz))
    with pytest.raises(com.UnsupportedOperationError, match="non-zero UTC offset"):
        con.compile(expr)


def test_string_schema_round_trip(con, temp_table):
    # `string` is created as VARCHAR(32672) and used to read back as
    # string(32672), which does not equal `string`.
    schema = ibis.schema({"a": "string", "b": "string(10)"})
    con.create_table(temp_table, schema=schema)
    try:
        assert con.table(temp_table).schema() == schema
        assert con.get_schema(temp_table) == schema
        assert con.sql(f'SELECT * FROM "{temp_table}"').schema() == schema
    finally:
        con.drop_table(temp_table, force=True)


@pytest.mark.parametrize("scale", [0, 3, 6])
def test_timestamp_scale_round_trip(con, temp_table, scale):
    # Scales above 6 are not covered: the driver returns microsecond datetimes,
    # so they read back as timestamp(6).
    schema = ibis.schema({"ts": f"timestamp({scale})"})
    con.create_table(temp_table, schema=schema)
    try:
        assert con.get_schema(temp_table) == schema
        assert con.table(temp_table).schema() == schema
        assert con.sql(f'SELECT * FROM "{temp_table}"').schema() == schema
    finally:
        con.drop_table(temp_table, force=True)


def test_sql_schema_keeps_bit_data_length(con, temp_table):
    # FOR BIT DATA is not ordinary character data, so it must not be normalised
    # to an unbounded string like VARCHAR(32672). The `string(32672)` asserted
    # here only preserves the current bounded-string reconstruction through
    # `con.sql()`. It does not claim that string is the right type for a binary
    # column: `get_schema()` maps it to binary, and that mismatch is a separate
    # issue.
    with con._safe_raw_sql(
        f'CREATE TABLE "{temp_table}" ("a" VARCHAR(32672),'
        ' "b" VARCHAR(32672) FOR BIT DATA)'
    ):
        pass
    try:
        schema = con.sql(f'SELECT * FROM "{temp_table}"').schema()
        assert schema["a"] == dt.string
        assert schema["b"] == dt.String(length=32672)
    finally:
        con.drop_table(temp_table, force=True)


@pytest.fixture
def join_tables(con, temp_table):
    other = gen_name("db2_test_other")
    for name in (temp_table, other):
        with con._safe_raw_sql(f'CREATE TABLE "{name}" ("id" INTEGER, "y" INTEGER)'):
            pass
        with con._safe_raw_sql(f'INSERT INTO "{name}" VALUES (1, 10), (2, 20)'):
            pass
    try:
        yield temp_table, other
    finally:
        con.drop_table(other, force=True)


@pytest.mark.parametrize(
    "query",
    [
        "SELECT 1 AS a, 2 AS a FROM SYSIBM.SYSDUMMY1",
        'SELECT a."id", b."id" FROM "{t1}" a JOIN "{t2}" b ON a."id" = b."id"',
        'SELECT * FROM "{t1}" a JOIN "{t2}" b ON a."id" = b."id"',
    ],
    ids=["same_alias", "self_join_labels", "star_over_join"],
)
def test_sql_duplicate_labels_raise(con, join_tables, query):
    # The schema would silently have fewer columns than the result, so
    # `execute()` failed with a confusing length mismatch and `to_pyarrow()`
    # dropped columns. Ask for an alias instead of renaming anything.
    t1, t2 = join_tables
    with pytest.raises(com.IntegrityError, match=r"Duplicate column name.*alias"):
        con.sql(query.format(t1=t1, t2=t2)).schema()


def test_sql_duplicate_labels_name_the_columns(con):
    with pytest.raises(com.IntegrityError, match=r"\['A'\]"):
        con.sql("SELECT 1 AS a, 2 AS a FROM SYSIBM.SYSDUMMY1").schema()


def test_sql_explicit_aliases_work(con, join_tables):
    t1, t2 = join_tables
    expr = con.sql(
        f'SELECT a."id" AS aid, b."id" AS bid FROM "{t1}" a'
        f' JOIN "{t2}" b ON a."id" = b."id"'
    )
    assert expr.columns == ("AID", "BID")
    assert len(expr.execute()) == 2


def test_sql_duplicate_labels_with_schema_override(con, join_tables):
    # An explicit schema skips inference, so the user's choice is used as is.
    t1, t2 = join_tables
    expr = con.sql(
        f'SELECT a."id", b."id" FROM "{t1}" a JOIN "{t2}" b ON a."id" = b."id"'
        " ORDER BY 1",
        schema={"x": "int32", "y": "int32"},
    )
    assert expr.columns == ("x", "y")
    assert expr.execute().values.tolist() == [[1, 1], [2, 2]]


def test_sql_unnamed_duplicate_expressions_work(con):
    # Db2 names unnamed columns `1` and `2`, so these are not duplicates.
    expr = con.sql("SELECT 1, 1 FROM SYSIBM.SYSDUMMY1")
    assert expr.columns == ("1", "2")
    assert expr.execute().values.tolist() == [[1, 1]]


def test_sql_labels_differing_only_by_case_are_distinct(con):
    expr = con.sql('SELECT 1 AS a, 2 AS "a" FROM SYSIBM.SYSDUMMY1')
    assert expr.columns == ("A", "a")
    assert expr.execute().values.tolist() == [[1, 2]]


def test_string_to_time_not_defined(con):
    # Do not silently ignore the supplied format.
    expr = ibis.literal("12:34:56").as_time("%H:%M:%S")
    with pytest.raises(com.OperationNotDefinedError, match="StringToTime"):
        con.compile(expr)


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        pytest.param([], [], id="empty"),
        pytest.param([None, None], [None, None], id="all-none"),
        pytest.param(["a", None], ["a", None], id="values"),
    ],
)
def test_create_table_dataframe_uses_explicit_schema(con, temp_table, data, expected):
    # Object columns with no values infer as the null type unless the supplied
    # schema is passed on when the DataFrame is wrapped.
    import pandas as pd

    schema = ibis.schema({"id": "int64", "name": "string"})
    df = pd.DataFrame(
        {
            "id": pd.Series(range(len(data)), dtype="int64"),
            "name": pd.Series(data, dtype=object),
        }
    )
    con.create_table(temp_table, df, schema=schema)
    try:
        assert con.table(temp_table).schema() == schema
        result = con.table(temp_table).order_by("id").execute()
        assert len(result) == len(expected)
        assert [None if pd.isna(v) else v for v in result["name"]] == expected
    finally:
        con.drop_table(temp_table, force=True)


def test_create_table_overwrite_view_leaves_no_staging_table(
    con, temp_table, monkeypatch
):
    # A fixed staging name makes a leftover table detectable.
    staging = gen_name("ibis_db2_table")
    monkeypatch.setattr(ibis.util, "gen_name", lambda _: staging)

    base = gen_name("db2_test_base")
    con.create_table(base, schema={"a": "int32"})
    con.create_view(temp_table, con.table(base))
    try:
        with pytest.raises(com.IbisInputError, match="is a view"):
            con.create_table(temp_table, schema={"a": "int32"}, overwrite=True)
        with pytest.raises(com.IbisInputError, match="is a view"):
            con.create_table(temp_table, con.table(base), overwrite=True, database=None)

        assert staging not in con.list_tables()
        # The view is still a view over the base table.
        assert temp_table in con.list_tables()
        assert con.table(temp_table).columns == ("a",)
        con.drop_view(temp_table)
        assert temp_table not in con.list_tables()
    finally:
        con.drop_view(temp_table, force=True)
        con.drop_table(staging, force=True)
        con.drop_table(base, force=True)
