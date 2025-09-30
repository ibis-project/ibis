"""Configuration for Beam backend tests."""

import pytest

import ibis
import ibis.expr.types as ir
from ibis.backends.beam import Backend


@pytest.fixture
def beam_backend():
    """Create a Beam backend for testing."""
    import apache_beam as beam
    
    # Create a test pipeline
    pipeline = beam.Pipeline()
    
    # Create the backend
    backend = Backend()
    backend.do_connect(pipeline)
    
    return backend


@pytest.fixture
def simple_table(beam_backend):
    """Create a simple test table."""
    import pandas as pd
    
    data = pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
        'age': [25, 30, 35, 40, 45],
        'salary': [50000, 60000, 70000, 80000, 90000]
    })
    
    # Create a temporary view
    table = beam_backend.create_view(
        'employees',
        data,
        temp=True
    )
    
    return table


@pytest.fixture
def empty_table(beam_backend):
    """Create an empty test table."""
    import pandas as pd
    
    data = pd.DataFrame({
        'id': [],
        'name': [],
        'age': [],
        'salary': []
    })
    
    # Create a temporary view
    table = beam_backend.create_view(
        'empty_employees',
        data,
        temp=True
    )
    
    return table
