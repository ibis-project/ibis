from __future__ import annotations

import pytest
from pytest import param

import ibis.expr.datatypes as dt
from ibis.backends.trino.datatypes import TrinoType

dtypes = [
    ("interval year to month", dt.Interval(unit="M")),
    ("interval day to second", dt.Interval(unit="ms")),
    ("bigint", dt.int64),
    ("boolean", dt.boolean),
    ("varbinary", dt.binary),
    ("double", dt.float64),
    ("real", dt.float32),
    ("smallint", dt.int16),
    ("timestamp", dt.Timestamp(scale=3)),
    ("timestamp(6)", dt.Timestamp(scale=6)),
    ("timestamp(9) with time zone", dt.Timestamp(scale=9, timezone="UTC")),
    ("timestamp with time zone", dt.Timestamp(scale=3, timezone="UTC")),
    ("date", dt.date),
    ("time", dt.time),
    ("time(6)", dt.time),
    ("time with time zone", dt.time),
    ("time(7) with time zone", dt.time),
    ("tinyint", dt.int8),
    ("integer", dt.int32),
    ("uuid", dt.uuid),
    ("char", dt.string),
    ("char(42)", dt.string),
    ("json", dt.json),
    ("ipaddress", dt.inet),
    ("varchar", dt.string),
    ("varchar(7)", dt.string),
    ("decimal", dt.Decimal(18, 3)),
    ("decimal(15, 0)", dt.Decimal(15, 0)),
    ("decimal(23, 5)", dt.Decimal(23, 5)),
    ("numeric", dt.Decimal(18, 3)),
    ("numeric(15, 0)", dt.Decimal(15, 0)),
    ("numeric(23, 5)", dt.Decimal(23, 5)),
    ("array(date)", dt.Array(dt.date)),
    ("array(array(date))", dt.Array(dt.Array(dt.date))),
    ("array(array(decimal(42, 23)))", dt.Array(dt.Array(dt.Decimal(42, 23)))),
    (
        "array(row(xYz map(varchar(3), double)))",
        dt.Array(dt.Struct(dict(xYz=dt.Map(dt.string, dt.float64)))),
    ),
    ("map(varchar, array(double))", dt.Map(dt.string, dt.Array(dt.float64))),
    (
        "row(a varchar, b array(tinyint), c map(bigint, row(d double)))",
        dt.Struct(
            dict(
                a=dt.string,
                b=dt.Array(dt.int8),
                c=dt.Map(dt.int64, dt.Struct(dict(d=dt.float64))),
            )
        ),
    ),
]


@pytest.mark.parametrize(
    ("trino_type", "ibis_type"),
    [param(trino_type, ibis_type, id=trino_type) for trino_type, ibis_type in dtypes],
)
def test_parse(trino_type, ibis_type):
    assert TrinoType.from_string(trino_type) == ibis_type
