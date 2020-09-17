import re

import numpy as np
import pandas as pd
import pytest
from pytest import param

import ibis

pytestmark = pytest.mark.omniscidb
pytest.importorskip('pymapd')


@pytest.mark.parametrize(
    ('result_fn', 'expected'),
    [
        param(
            lambda t: t[t, ibis.literal(1).degrees().name('n')].limit(1)['n'],
            57.2957795130823,
            id='literal_degree',
        ),
        param(
            lambda t: t[t, ibis.literal(1).radians().name('n')].limit(1)['n'],
            0.0174532925199433,
            id='literal_radians',
        ),
        param(
            lambda t: t.double_col.corr(t.float_col),
            1.000000000000113,
            id='double_float_correlation',
        ),
        param(
            lambda t: t.double_col.cov(t.float_col),
            91.67005567565313,
            id='double_float_covariance',
        ),
    ],
)
def test_operations_scalar(alltypes, result_fn, expected):
    result = result_fn(alltypes).execute()
    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize(
    ('result_fn', 'check_result'),
    [
        param(
            lambda t: (
                t[t.date_string_col][t.date_string_col.ilike('10/%')].limit(1)
            ),
            lambda v: v.startswith('10/'),
            id='string_ilike',
        )
    ],
)
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
        t2.teamID,
    ]
    k = [t1, t2.teamID]
    df = (
        t1.left_join(
            t2,
            (
                (t1.yearID == t2.yID)
                & (t1.playerID == t2.pID)
                & (t1.lgID == t2.lID)
            ),
        )[k]
        .materialize()
        .execute()
    )
    assert df.size == 80


def test_cross_join(alltypes):
    d = alltypes.double_col

    tier = d.histogram(10).name('hist_bin')
    expr = (
        alltypes.group_by(tier)
        .aggregate([d.min(), d.max(), alltypes.count()])
        .sort_by('hist_bin')
    )
    df = expr.execute()
    assert df.size == 40
    assert df['count'][0] == 730


def test_where_operator(alltypes):
    t = alltypes.sort_by('index').limit(10)
    expr = ibis.where(t.index > 4, 1, 0)
    counts = expr.execute().value_counts()
    assert counts[0] == 5
    assert counts[1] == 5


@pytest.mark.parametrize('name', ['regular_name', 'star_name*', 'space_name '])
def test_quote_name(alltypes, name):
    expr = alltypes.aggregate(alltypes.count().name(name))
    assert name in expr.execute()


def test_timestamp_col(alltypes):
    # https://github.com/ibis-project/ibis/issues/1613
    alltypes[alltypes.timestamp_col < '2000-03-01'].execute()


@pytest.mark.parametrize(
    ('result_fn', 'expected_fn'),
    [
        param(
            lambda t: t.double_col.arbitrary(),
            lambda t: t.double_col.iloc[-1],
            id='double_col_arbitrary_none',
        )
    ],
)
def test_arbitrary_none(alltypes, df_alltypes, result_fn, expected_fn):
    expr = result_fn(alltypes)
    result = expr.execute()
    expected = expected_fn(df_alltypes)
    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize(
    ('ibis_op', 'sql_op'),
    [('sum', 'sum'), ('mean', 'avg'), ('max', 'max'), ('min', 'min')],
)
def test_agg_with_bool(alltypes, ibis_op, sql_op):
    regex = re.compile(r'\s{2}|\n')

    expr = getattr(alltypes.bool_col, ibis_op)()
    sql_check = (
        'SELECT {}(CASE WHEN "bool_col" THEN 1 ELSE 0 END) AS "{}"'
        'FROM functional_alltypes'
    ).format(sql_op, ibis_op)

    assert regex.sub('', expr.compile()) == regex.sub('', sql_check)


@pytest.mark.parametrize(
    'expr_fn',
    [
        lambda t: t.float_col.mean(where=t.date_string_col == '11/01/10'),
        lambda t: t.float_col.bucket([0, 1, 3]).name('bucket'),
    ],
)
def test_expr_with_null_literal(alltypes, expr_fn):
    expr_fn(alltypes).execute()
