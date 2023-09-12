from __future__ import annotations

import pytest
import sqlalchemy as sa
from pytest import param

import ibis
import ibis.expr.datatypes as dt
from ibis import udf
from ibis.backends.base.sql.alchemy.geospatial import geospatial_supported

DB_TYPES = [
    # Exact numbers
    ("BIGINT", dt.int64),
    ("BIT", dt.boolean),
    ("DECIMAL", dt.Decimal(precision=18, scale=0)),
    ("DECIMAL(5, 2)", dt.Decimal(precision=5, scale=2)),
    ("INT", dt.int32),
    ("MONEY", dt.int64),
    ("NUMERIC", dt.Decimal(18, 0)),
    ("NUMERIC(10,5)", dt.Decimal(10, 5)),
    ("NUMERIC(14,3)", dt.Decimal(14, 3)),
    ("SMALLINT", dt.int16),
    ("SMALLMONEY", dt.int32),
    ("TINYINT", dt.int8),
    # Approximate numerics
    ("REAL", dt.float32),
    ("FLOAT", dt.float64),
    ("FLOAT(3)", dt.float32),
    ("FLOAT(25)", dt.float64),
    # Date and time
    ("DATE", dt.date),
    ("TIME", dt.time),
    ("DATETIME2", dt.timestamp(scale=7)),
    ("DATETIMEOFFSET", dt.timestamp(scale=7, timezone="UTC")),
    ("SMALLDATETIME", dt.timestamp),
    ("DATETIME", dt.timestamp),
    # Characters strings
    ("CHAR", dt.string),
    ("TEXT", dt.string),
    ("VARCHAR", dt.string),
    # Unicode character strings
    ("NCHAR", dt.string),
    ("NTEXT", dt.string),
    ("NVARCHAR", dt.string),
    # Binary strings
    ("BINARY", dt.binary),
    ("VARBINARY", dt.binary),
    ("IMAGE", dt.binary),
    # Other data types
    ("UNIQUEIDENTIFIER", dt.uuid),
    ("TIMESTAMP", dt.binary(nullable=False)),
]


skipif_no_geospatial_deps = pytest.mark.skipif(
    not geospatial_supported, reason="geospatial dependencies not installed"
)

broken_sqlalchemy_autoload = pytest.mark.xfail(
    reason="scale not inferred by sqlalchemy autoload"
)


@pytest.mark.parametrize(
    ("server_type", "expected_type"),
    DB_TYPES
    + [
        param("GEOMETRY", dt.geometry, marks=[skipif_no_geospatial_deps]),
        param("GEOGRAPHY", dt.geography, marks=[skipif_no_geospatial_deps]),
    ]
    + [
        param(
            "DATETIME2(4)", dt.timestamp(scale=4), marks=[broken_sqlalchemy_autoload]
        ),
        param(
            "DATETIMEOFFSET(5)",
            dt.timestamp(scale=5, timezone="UTC"),
            marks=[broken_sqlalchemy_autoload],
        ),
    ],
    ids=str,
)
def test_get_schema_from_query(con, server_type, expected_type, temp_table):
    expected_schema = ibis.schema(dict(x=expected_type))
    with con.begin() as c:
        c.exec_driver_sql(f"CREATE TABLE [{temp_table}] (x {server_type})")
    expected_schema = ibis.schema(dict(x=expected_type))
    result_schema = con._get_schema_using_query(f"SELECT * FROM [{temp_table}]")
    assert result_schema == expected_schema
    t = con.table(temp_table)
    assert t.schema() == expected_schema


def test_builtin_scalar_udf(con):
    @udf.scalar.builtin
    def difference(a: str, b: str) -> int:
        """Soundex difference between two strings."""

    expr = difference("foo", "moo")
    result = con.execute(expr)
    assert result == 3


def test_builtin_agg_udf(con):
    @udf.agg.builtin
    def count_big(x) -> int:
        """The biggest of counts."""

    ft = con.tables.functional_alltypes
    expr = count_big(ft.id)
    assert expr.execute() == ft.count().execute()


def test_builtin_agg_udf_filtered(con):
    @udf.agg.builtin
    def count_big(x, where: bool = True) -> int:
        """The biggest of counts."""

    ft = con.tables.functional_alltypes
    expr = count_big(ft.id)
    with pytest.raises(
        sa.exc.OperationalError, match="An expression of non-boolean type specified"
    ):
        assert expr.execute()

    expr = count_big(ft.id, where=ft.id == 1)
    assert expr.execute() == ft[ft.id == 1].count().execute()
