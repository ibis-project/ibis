from __future__ import annotations

import hypothesis as h
import hypothesis.strategies as st
import numpy as np
import pytest

import ibis
import ibis.expr.datatypes as dt
import ibis.expr.schema as sch
import ibis.expr.types as ir
import ibis.tests.strategies as its
from ibis.common.annotations import ValidationError


@h.given(its.null_dtype)
def test_null_dtype(dtype):
    assert isinstance(dtype, dt.Null)
    assert dtype.is_null() is True
    assert dtype.nullable is True


@h.given(its.boolean_dtype())
def test_boolean_dtype(dtype):
    assert isinstance(dtype, dt.Boolean)
    assert dtype.is_boolean() is True


@h.given(its.signed_integer_dtypes())
def test_signed_integer_dtype(dtype):
    assert isinstance(dtype, dt.SignedInteger)
    assert dtype.is_integer() is True


@h.given(its.unsigned_integer_dtypes())
def test_unsigned_integer_dtype(dtype):
    assert isinstance(dtype, dt.UnsignedInteger)
    assert dtype.is_integer() is True


@h.given(its.floating_dtypes())
def test_floating_dtype(dtype):
    assert isinstance(dtype, dt.Floating)
    assert dtype.is_floating() is True


@h.given(its.numeric_dtypes())
def test_numeric_dtype(dtype):
    assert isinstance(dtype, dt.Numeric)
    assert dtype.is_numeric() is True


@h.given(its.numeric_dtypes(nullable=st.just(True)))
def test_numeric_dtypes_nullable(dtype):
    assert dtype.nullable is True
    assert dtype.is_numeric() is True


@h.given(its.numeric_dtypes(nullable=st.just(False)))
def test_numeric_dtypes_non_nullable(dtype):
    assert dtype.nullable is False
    assert dtype.is_numeric() is True


@h.given(its.timestamp_dtype())
def test_timestamp_dtype(dtype):
    assert isinstance(dtype, dt.Timestamp)
    assert isinstance(dtype.timezone, (type(None), str))
    assert dtype.is_timestamp() is True


@h.given(its.interval_dtype())
def test_interval_dtype(dtype):
    assert isinstance(dtype, dt.Interval)
    assert dtype.is_interval() is True


@h.given(its.temporal_dtypes())
def test_temporal_dtype(dtype):
    assert isinstance(dtype, dt.Temporal)
    assert dtype.is_temporal() is True


@h.given(its.primitive_dtypes())
def test_primitive_dtype(dtype):
    assert isinstance(dtype, dt.Primitive)
    assert dtype.is_primitive() is True


@h.given(its.geospatial_dtypes())
def test_geospatial_dtype(dtype):
    assert isinstance(dtype, dt.GeoSpatial)
    assert dtype.is_geospatial() is True


@h.given(its.array_dtypes(its.primitive_dtypes()))
def test_array_dtype(dtype):
    assert isinstance(dtype, dt.Array)
    assert isinstance(dtype.value_type, dt.Primitive)
    assert dtype.is_array() is True


@h.given(its.array_dtypes(its.array_dtypes(its.primitive_dtypes())))
def test_array_array_dtype(dtype):
    assert isinstance(dtype, dt.Array)
    assert isinstance(dtype.value_type, dt.Array)
    assert isinstance(dtype.value_type.value_type, dt.Primitive)


@h.given(its.map_dtypes(its.primitive_dtypes(), its.boolean_dtype()))
def test_map_dtype(dtype):
    assert isinstance(dtype, dt.Map)
    assert isinstance(dtype.key_type, dt.Primitive)
    assert isinstance(dtype.value_type, dt.Boolean)
    assert dtype.is_map() is True


@h.given(its.struct_dtypes())
def test_struct_dtype(dtype):
    assert isinstance(dtype, dt.Struct)
    assert all(t.is_primitive() for t in dtype.types)
    assert dtype.is_struct() is True


@h.given(its.struct_dtypes(its.variadic_dtypes()))
def test_struct_variadic_dtype(dtype):
    assert isinstance(dtype, dt.Struct)
    assert all(t.is_variadic() for t in dtype.types)
    assert dtype.is_struct() is True


@h.given(its.variadic_dtypes())
def test_variadic_dtype(dtype):
    assert isinstance(dtype, dt.Variadic)
    assert dtype.is_variadic() is True


@h.given(its.all_dtypes())
def test_all_dtypes(dtype):
    assert isinstance(dtype, dt.DataType)


@h.given(its.schema())
def test_schema(schema):
    assert isinstance(schema, sch.Schema)
    assert all(t.is_primitive() for t in schema.types)
    assert all(isinstance(n, str) for n in schema.names)
    assert len(set(schema.names)) == len(schema.names)


@h.given(its.schema(its.array_dtypes(its.numeric_dtypes())))
def test_schema_array_dtype(schema):
    assert isinstance(schema, sch.Schema)
    assert all(t.is_array() for t in schema.types)
    assert all(isinstance(n, str) for n in schema.names)


@h.given(its.primitive_dtypes())
def test_primitive_dtypes_to_pandas(dtype):
    assert isinstance(dtype.to_pandas(), np.dtype)


@h.given(its.schema())
def test_schema_to_pandas(schema):
    pandas_schema = schema.to_pandas()
    assert len(pandas_schema) == len(schema)


@h.given(its.memtable(its.schema(its.integer_dtypes(), max_size=5)))
def test_memtable(memtable):
    assert isinstance(memtable, ir.TableExpr)
    assert isinstance(memtable.schema(), sch.Schema)


@h.given(its.all_dtypes())
def test_deferred_literal(dtype):
    with pytest.raises(ValidationError):
        ibis.literal(ibis._.a, type=dtype)


# TODO(kszucs): we enforce field name uniqueness in the schema, but we don't for Struct datatype
