"""Tests for Beam SQL datatypes."""

import pytest

import ibis.expr.datatypes as dt
from ibis.backends.beam.datatypes import BeamType


def test_beam_type_to_ibis():
    """Test conversion from Beam types to Ibis types."""
    # Test basic types
    assert isinstance(BeamType.to_ibis("VARCHAR"), dt.String)
    assert isinstance(BeamType.to_ibis("BOOLEAN"), dt.Boolean)
    assert isinstance(BeamType.to_ibis("INTEGER"), dt.Int32)
    assert isinstance(BeamType.to_ibis("BIGINT"), dt.Int64)
    assert isinstance(BeamType.to_ibis("FLOAT"), dt.Float32)
    assert isinstance(BeamType.to_ibis("DOUBLE"), dt.Float64)
    assert isinstance(BeamType.to_ibis("DATE"), dt.Date)
    assert isinstance(BeamType.to_ibis("TIME"), dt.Time)
    assert isinstance(BeamType.to_ibis("TIMESTAMP"), dt.Timestamp)


def test_beam_type_from_ibis():
    """Test conversion from Ibis types to Beam types."""
    # Test basic types
    assert BeamType.from_ibis(dt.String()) == "VARCHAR"
    assert BeamType.from_ibis(dt.Boolean()) == "BOOLEAN"
    assert BeamType.from_ibis(dt.Int32()) == "INTEGER"
    assert BeamType.from_ibis(dt.Int64()) == "BIGINT"
    assert BeamType.from_ibis(dt.Float32()) == "FLOAT"
    assert BeamType.from_ibis(dt.Float64()) == "DOUBLE"
    assert BeamType.from_ibis(dt.Date()) == "DATE"
    assert BeamType.from_ibis(dt.Time()) == "TIME"
    assert BeamType.from_ibis(dt.Timestamp()) == "TIMESTAMP(6)"


def test_beam_type_timestamp_precision():
    """Test timestamp precision handling."""
    # Test with precision
    assert BeamType.from_ibis(dt.Timestamp(scale=3)) == "TIMESTAMP(3)"
    
    # Test without precision (default)
    assert BeamType.from_ibis(dt.Timestamp()) == "TIMESTAMP(6)"


def test_beam_type_array():
    """Test array type handling."""
    # Test array type
    array_type = dt.Array(dt.String())
    beam_type = BeamType.from_ibis(array_type)
    assert beam_type == "ARRAY<VARCHAR>"
    
    # Test nested array
    nested_array = dt.Array(dt.Array(dt.Int32()))
    beam_type = BeamType.from_ibis(nested_array)
    assert beam_type == "ARRAY<ARRAY<INTEGER>>"


def test_beam_type_map():
    """Test map type handling."""
    # Test map type
    map_type = dt.Map(dt.String(), dt.Int32())
    beam_type = BeamType.from_ibis(map_type)
    assert beam_type == "MAP<VARCHAR, INTEGER>"


def test_beam_type_struct():
    """Test struct type handling."""
    # Test struct type
    struct_type = dt.Struct({
        'name': dt.String(),
        'age': dt.Int32(),
        'active': dt.Boolean()
    })
    beam_type = BeamType.from_ibis(struct_type)
    assert 'name VARCHAR' in beam_type
    assert 'age INTEGER' in beam_type
    assert 'active BOOLEAN' in beam_type
    assert beam_type.startswith('ROW(')
    assert beam_type.endswith(')')


def test_beam_type_nullable():
    """Test nullable type handling."""
    # Test nullable types
    nullable_string = dt.String(nullable=True)
    beam_type = BeamType.from_ibis(nullable_string)
    assert beam_type == "VARCHAR"
    
    # Test non-nullable types
    non_nullable_string = dt.String(nullable=False)
    beam_type = BeamType.from_ibis(non_nullable_string)
    assert beam_type == "VARCHAR NOT NULL"
