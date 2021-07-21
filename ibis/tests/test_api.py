import pytest

import ibis
from ibis.backends.base import BaseBackend


def test_top_level_api():
    known_api = [
        'Expr',
        'IbisError',
        'NA',
        'Schema',
        'aggregate',
        'array',
        'api',
        'case',
        'cast',
        'coalesce',
        'cross_join',
        'cumulative_window',
        'date',
        'desc',
        'expr_list',
        'geo_area',
        'geo_as_binary',
        'geo_as_ewkb',
        'geo_as_ewkt',
        'geo_as_text',
        'geo_azimuth',
        'geo_buffer',
        'geo_centroid',
        'geo_contains',
        'geo_contains_properly',
        'geo_covered_by',
        'geo_covers',
        'geo_crosses',
        'geo_d_fully_within',
        'geo_d_within',
        'geo_difference',
        'geo_disjoint',
        'geo_distance',
        'geo_end_point',
        'geo_envelope',
        'geo_equals',
        'geo_geometry_n',
        'geo_geometry_type',
        'geo_intersection',
        'geo_intersects',
        'geo_is_valid',
        'geo_length',
        'geo_line_locate_point',
        'geo_line_merge',
        'geo_line_substring',
        'geo_max_distance',
        'geo_n_points',
        'geo_n_rings',
        'geo_ordering_equals',
        'geo_overlaps',
        'geo_perimeter',
        'geo_point',
        'geo_point_n',
        'geo_simplify',
        'geo_srid',
        'geo_start_point',
        'geo_touches',
        'geo_transform',
        'geo_unary_union',
        'geo_union',
        'geo_within',
        'geo_x',
        'geo_x_max',
        'geo_x_min',
        'geo_y',
        'geo_y_max',
        'geo_y_min',
        'greatest',
        'ifelse',
        'infer_dtype',
        'infer_schema',
        'interval',
        'ir',
        'join',
        'least',
        'literal',
        'negate',
        'now',
        'null',
        'options',
        'param',
        'pi',
        'prevent_rewrite',
        'random',
        'range_window',
        'row_number',
        'rows_with_max_lookback',
        'schema',
        'sequence',
        'table',
        'time',
        'timestamp',
        'trailing_range_window',
        'trailing_window',
        'util',
        'where',
        'window',
    ]

    assert sorted(ibis.__all__) == sorted(known_api)


def test_backends_are_cached():
    # can't use `hasattr` since it calls `__getattr__`
    if 'sqlite' in dir(ibis):
        del ibis.sqlite
    assert isinstance(ibis.sqlite, BaseBackend)
    assert 'sqlite' in dir(ibis)


def test_missing_backend():
    msg = "If you are trying to access the 'foo' backend"
    with pytest.raises(AttributeError, match=msg):
        ibis.foo


def test_multiple_backends(mocker):
    mocker.patch(
        'pkg_resources.iter_entry_points',
        return_value=['backend1', 'backend2', 'backend3'],
    )

    msg = "3 packages found for backend 'foo'"
    with pytest.raises(RuntimeError, match=msg):
        ibis.foo
