import ibis
import numpy as np
import pandas as pd
import pytest

from pytest import param

pytestmark = pytest.mark.mapd
pytest.importorskip('pymapd')


@pytest.mark.parametrize(('result_fn', 'expected'), [
    param(
        lambda t: t[t, ibis.literal(1).degrees().name('n')].limit(1)['n'],
        57.2957795130823,
        id='literal_degree'
    ),
    param(
        lambda t: t[t, ibis.literal(1).radians().name('n')].limit(1)['n'],
        0.0174532925199433,
        id='literal_radians'
    ),
    param(
        lambda t: t.double_col.corr(t.float_col),
        1.000000000000113,
        id='double_float_correlation'
    ),
    param(
        lambda t: t.double_col.cov(t.float_col),
        91.67005567565313,
        id='double_float_covariance'
    )
])
def test_operations_scalar(alltypes, result_fn, expected):
    result = result_fn(alltypes).execute()
    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize(('result_fn', 'check_result'), [
    param(
        lambda t: (
            t[t.date_string_col][t.date_string_col.ilike('10/%')].limit(1)
        ),
        lambda v: v.startswith('10/'),
        id='string_ilike'
    )
])
def test_string_operations(alltypes, result_fn, check_result):
    result = result_fn(alltypes).execute()

    if isinstance(result, pd.DataFrame):
        result = result.values[0][0]
    assert check_result(result)


def test_join_diff_name(awards_players, batting):
    """Test left join operation between columns with different name"""
    t1 = awards_players.sort_by('yearID').limit(10)
    t2 = batting.sort_by('yearID').limit(10)
    t2 = t2[
        t2.playerID.name('pID'),
        t2.yearID.name('yID'),
        t2.lgID.name('lID'),
        t2.teamID
    ]
    k = [t1, t2.teamID]
    df = t1.left_join(
        t2, (
                (t1.yearID == t2.yID) &
                (t1.playerID == t2.pID) &
                (t1.lgID == t2.lID)
        )
    )[k].materialize().execute()
    assert df.size == 80


def test_cross_join(alltypes):
    d = alltypes.double_col

    tier = d.histogram(10).name('hist_bin')
    expr = (alltypes.group_by(tier)
            .aggregate([d.min(), d.max(), alltypes.count()])
            .sort_by('hist_bin'))
    df = expr.execute()
    assert df.size == 40
    assert df['count'][0] == 730


def test_where_operator(alltypes):
    t = alltypes.sort_by('index').limit(10)
    expr = ibis.where(t.index > 4, 1, 0)
    counts = expr.execute().value_counts()
    assert counts[0] == 5
    assert counts[1] == 5


@pytest.mark.parametrize('name', [
    'regular_name',
    'star_name*',
    'space_name ',
])
def test_quote_name(alltypes, name):
    expr = alltypes.aggregate(alltypes.count().name(name))
    assert name in expr.execute()


def test_timestamp_col(alltypes):
    # https://github.com/ibis-project/ibis/issues/1613
    alltypes[alltypes.timestamp_col < '2000-03-01'].execute()


def test_literal_geospatial():
    # point
    point_0 = (0, 0)
    point = ibis.literal(point_0, type='point')
    assert ibis.mapd.compile(point) == "SELECT 'POINT(0 0)' AS tmp"

    # line
    line_0 = [point_0, point_0]
    line = ibis.literal(line_0, type='linestring')
    assert ibis.mapd.compile(line) == "SELECT 'LINESTRING(0 0, 0 0)' AS tmp"

    # polygon
    polygon_0 = [tuple(line_0), tuple(line_0)]
    polygon = ibis.literal(polygon_0, type='polygon')
    assert ibis.mapd.compile(polygon) == (
        "SELECT 'POLYGON((0 0, 0 0), (0 0, 0 0))' AS tmp"
    )

    # multipolygon
    mpolygon_0 = [tuple(polygon_0), tuple(polygon_0)]
    mpolygon = ibis.literal(mpolygon_0, type='multipolygon')
    assert ibis.mapd.compile(mpolygon) == (
        "SELECT 'MULTIPOLYGON(((0 0, 0 0), (0 0, 0 0)), "
        "((0 0, 0 0), (0 0, 0 0)))' AS tmp"
    )


@pytest.mark.parametrize(('result_fn', 'expected_fn'), [
    param(
        lambda t: t.double_col.arbitrary(),
        lambda t: t.double_col.iloc[-1],
        id='double_col_arbitrary_none'
    ),
])
def test_arbitrary_none(alltypes, df_alltypes, result_fn, expected_fn):
    expr = result_fn(alltypes)
    result = expr.execute()
    expected = expected_fn(df_alltypes)
    np.testing.assert_allclose(result, expected)
