"""Tests for the Beam SQL compiler."""

import pytest

import ibis
from ibis.backends.beam import Backend


def test_beam_compiler_basic():
    """Test basic Beam SQL compilation."""
    import apache_beam as beam
    
    pipeline = beam.Pipeline()
    backend = Backend()
    backend.do_connect(pipeline)
    
    # Test basic SQL compilation
    expr = ibis.literal(1)
    sql = backend.compile(expr)
    assert sql is not None


def test_beam_compiler_aggregation():
    """Test Beam SQL aggregation compilation."""
    import apache_beam as beam
    import pandas as pd
    
    pipeline = beam.Pipeline()
    backend = Backend()
    backend.do_connect(pipeline)
    
    # Create test data
    data = pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'value': [10, 20, 30, 40, 50]
    })
    
    # Create table
    table = backend.create_view('test_table', data, temp=True)
    
    # Test aggregation
    expr = table.value.sum()
    sql = backend.compile(expr)
    assert sql is not None
    assert 'SUM' in sql.upper()


def test_beam_compiler_filter():
    """Test Beam SQL filter compilation."""
    import apache_beam as beam
    import pandas as pd
    
    pipeline = beam.Pipeline()
    backend = Backend()
    backend.do_connect(pipeline)
    
    # Create test data
    data = pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'value': [10, 20, 30, 40, 50]
    })
    
    # Create table
    table = backend.create_view('test_table', data, temp=True)
    
    # Test filter
    expr = table.filter(table.value > 20)
    sql = backend.compile(expr)
    assert sql is not None
    assert 'WHERE' in sql.upper()


def test_beam_compiler_join():
    """Test Beam SQL join compilation."""
    import apache_beam as beam
    import pandas as pd
    
    pipeline = beam.Pipeline()
    backend = Backend()
    backend.do_connect(pipeline)
    
    # Create test data
    data1 = pd.DataFrame({
        'id': [1, 2, 3],
        'name': ['A', 'B', 'C']
    })
    
    data2 = pd.DataFrame({
        'id': [1, 2, 4],
        'value': [100, 200, 400]
    })
    
    # Create tables
    table1 = backend.create_view('table1', data1, temp=True)
    table2 = backend.create_view('table2', data2, temp=True)
    
    # Test join
    expr = table1.join(table2, 'id')
    sql = backend.compile(expr)
    assert sql is not None
    assert 'JOIN' in sql.upper()


def test_beam_compiler_groupby():
    """Test Beam SQL groupby compilation."""
    import apache_beam as beam
    import pandas as pd
    
    pipeline = beam.Pipeline()
    backend = Backend()
    backend.do_connect(pipeline)
    
    # Create test data
    data = pd.DataFrame({
        'category': ['A', 'A', 'B', 'B', 'C'],
        'value': [10, 20, 30, 40, 50]
    })
    
    # Create table
    table = backend.create_view('test_table', data, temp=True)
    
    # Test groupby
    expr = table.group_by('category').agg(value_sum=table.value.sum())
    sql = backend.compile(expr)
    assert sql is not None
    assert 'GROUP BY' in sql.upper()
    assert 'SUM' in sql.upper()
