from __future__ import annotations

import pytest

import ibis
import ibis.expr.types as ir
from ibis.backends.conftest import TEST_TABLES


@pytest.fixture
def simple_schema():
    return [
        ('a', 'int8'),
        ('b', 'int16'),
        ('c', 'int32'),
        ('d', 'int64'),
        ('e', 'float32'),
        ('f', 'float64'),
        ('g', 'string'),
        ('h', 'boolean'),
        ('i', 'timestamp'),
        ('j', 'date'),
        ('k', 'time'),
    ]


@pytest.fixture
def simple_table(simple_schema):
    return ibis.table(simple_schema, name='table')


@pytest.fixture
def batting() -> ir.Table:
    return ibis.table(schema=TEST_TABLES["batting"], name="batting")


@pytest.fixture
def awards_players() -> ir.Table:
    return ibis.table(schema=TEST_TABLES["awards_players"], name="awards_players")
