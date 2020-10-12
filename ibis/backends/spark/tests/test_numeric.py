import pytest

import ibis

pytestmark = pytest.mark.spark


def test_double_format(client):
    expr = ibis.literal(1.0)
    assert client.compile(expr) == 'SELECT 1.0d AS `tmp`'


def test_float_format(client):
    expr = ibis.literal(1.0, type='float')
    assert client.compile(expr) == 'SELECT CAST(1.0 AS FLOAT) AS `tmp`'
