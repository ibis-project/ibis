from __future__ import annotations

import pytest

import ibis


@pytest.fixture
def geo_table():
    return ibis.table({"geo1": "geometry", "geo2": "geometry"}, name="t")


def test_geospatial_unary_op_repr(geo_table):
    expr = geo_table.geo1.centroid()
    assert expr.op().name in str(expr)


def test_geospatial_bin_op_repr(geo_table):
    expr = geo_table.geo1.d_within(geo_table.geo2, 3.0)
    assert expr.op().name in str(expr)
    assert "distance=" in str(expr)


def test_geospatial_bin_op_repr_no_kwarg(geo_table):
    expr = geo_table.geo1.distance(geo_table.geo2)
    assert expr.op().name in str(expr)
    assert "distance=" not in str(expr)
    # test that there isn't a trailing comma from empty kwargs
    assert str(expr).endswith("r0.geo2)")
