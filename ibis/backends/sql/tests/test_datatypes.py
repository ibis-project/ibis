from __future__ import annotations

import hypothesis as h
import hypothesis.strategies as st
import pytest
import sqlglot.expressions as sge

import ibis
import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.tests.strategies as its
from ibis.backends.sql.datatypes import (
    ClickHouseType,
    DuckDBType,
    PostgresType,
    SqlglotType,
)


def assert_dtype_roundtrip(ibis_type, sqlglot_expected=None):
    sqlglot_result = SqlglotType.from_ibis(ibis_type)
    assert isinstance(sqlglot_result, sge.DataType)
    if sqlglot_expected is not None:
        assert sqlglot_result == sqlglot_expected

    restored_dtype = SqlglotType.to_ibis(sqlglot_result)
    assert ibis_type == restored_dtype


true = st.just(True)

roundtripable_types = st.deferred(
    lambda: (
        its.null_dtype
        | its.boolean_dtype(nullable=true)
        | its.integer_dtypes(nullable=true)
        | st.just(dt.Float32(nullable=True))
        | st.just(dt.Float64(nullable=True))
        | its.string_dtype(nullable=true)
        | its.binary_dtype(nullable=true)
        | its.json_dtype(nullable=true)
        | its.inet_dtype(nullable=true)
        | its.uuid_dtype(nullable=true)
        | its.date_dtype(nullable=true)
        | its.time_dtype(nullable=true)
        | its.timestamp_dtype(timezone=st.none(), nullable=true)
        | its.array_dtypes(roundtripable_types, nullable=true)
        | its.map_dtypes(roundtripable_types, roundtripable_types, nullable=true)
        | its.struct_dtypes(roundtripable_types, nullable=true)
        | its.geospatial_dtypes(nullable=true)
        | its.decimal_dtypes(nullable=true)
        | its.interval_dtype(nullable=true)
    )
)

# not roundtrippable:
# - float16
# - macaddr
# - interval?


@h.given(roundtripable_types)
def test_roundtripable_types(ibis_type):
    assert_dtype_roundtrip(ibis_type)


def test_interval_without_unit():
    with pytest.raises(com.IbisTypeError, match="precision is None"):
        SqlglotType.from_string("INTERVAL")
    assert PostgresType.from_string("INTERVAL") == dt.Interval("s")
    assert DuckDBType.from_string("INTERVAL") == dt.Interval("us")


@pytest.mark.parametrize(
    "sgetyp",
    [
        sge.DataType(this=sge.DataType.Type.UINT256),
        sge.DataType(this=sge.DataType.Type.UINT128),
        sge.DataType(this=sge.DataType.Type.BIGSERIAL),
        sge.DataType(this=sge.DataType.Type.HLLSKETCH),
        sge.DataType(this=sge.DataType.Type.USERDEFINED, kind='"MySchema"."MyEnum"'),
    ],
)
@pytest.mark.parametrize(
    "typengine",
    [ClickHouseType, PostgresType, DuckDBType],
)
def test_unsupported_dtypes_are_unknown(typengine, sgetyp):
    ibis_type = typengine.to_ibis(sgetyp)
    assert ibis_type.is_unknown()
    assert ibis_type.raw_type == sgetyp


@pytest.mark.parametrize(
    "s,parsed",
    [
        ("VARCHAR", dt.String()),
        ("VECTOR", dt.Unknown(sge.DataType(this=sge.DataType.Type.VECTOR))),
        (
            '"MySchema"."MyEnum"',
            dt.Unknown(
                sge.DataType(
                    this=sge.DataType.Type.USERDEFINED, kind='"MySchema"."MyEnum"'
                )
            ),
        ),
    ],
)
@pytest.mark.parametrize(
    "typengine",
    [ClickHouseType, PostgresType, DuckDBType],
)
def test_from_string(typengine, s, parsed):
    ibis_type = typengine.from_string(s)
    # different backends have different default nullability, normalize to True
    ibis_type = ibis_type.copy(nullable=True)
    assert ibis_type == parsed


def test_cast_to_unknown():
    dtype = dt.Unknown(
        sge.DataType(this=sge.DataType.Type.USERDEFINED, kind='"MySchema"."MyEnum"')
    )
    e = ibis.literal(4).cast(dtype)
    sql = ibis.to_sql(e)
    assert """CAST(4 AS "MySchema"."MyEnum")""" in sql


def test_unknown_repr():
    dtype = dt.Unknown(
        sge.DataType(this=sge.DataType.Type.USERDEFINED, kind='"MySchema"."MyEnum"')
    )
    result = str(dtype)
    expected = 'unknown(DataType(this=Type.USERDEFINED, kind="MySchema"."MyEnum"))'
    assert result == expected
