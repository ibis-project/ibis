from __future__ import annotations

import hypothesis as h
import hypothesis.strategies as st
import pytest
import sqlglot.expressions as sge

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
    "typ",
    [
        sge.DataType.Type.UINT256,
        sge.DataType.Type.UINT128,
        sge.DataType.Type.BIGSERIAL,
        sge.DataType.Type.HLLSKETCH,
    ],
)
@pytest.mark.parametrize(
    "typengine",
    [ClickHouseType, PostgresType, DuckDBType],
)
def test_unsupported_dtypes_are_unknown(typengine, typ):
    assert typengine.to_ibis(sge.DataType(this=typ)) == dt.unknown
