from __future__ import annotations

import hypothesis as h
import hypothesis.strategies as st
import sqlglot.expressions as sge

import ibis.expr.datatypes as dt
import ibis.tests.strategies as its
from ibis.backends.base.sqlglot.datatypes import SqlglotType


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
    )
)

# not roundtrippable:
# - float16
# - macaddr
# - interval?


@h.given(roundtripable_types)
def test_roundtripable_types(ibis_type):
    assert_dtype_roundtrip(ibis_type)
