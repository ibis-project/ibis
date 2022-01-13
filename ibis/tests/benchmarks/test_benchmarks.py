import numpy as np
import pandas as pd
import pytest

import ibis
import ibis.expr.datatypes as dt
from ibis.backends.pandas.udf import udf


def make_t():
    return ibis.table(
        [
            ('_timestamp', 'int32'),
            ('dim1', 'int32'),
            ('dim2', 'int32'),
            ('valid_seconds', 'int32'),
            ('meas1', 'int32'),
            ('meas2', 'int32'),
            ('year', 'int32'),
            ('month', 'int32'),
            ('day', 'int32'),
            ('hour', 'int32'),
            ('minute', 'int32'),
        ],
        name="t",
    )


@pytest.fixture
def t():
    return make_t()


def make_base(t):
    return (
        (t.year > 2016)
        | ((t.year == 2016) & (t.month > 6))
        | ((t.year == 2016) & (t.month == 6) & (t.day > 6))
        | ((t.year == 2016) & (t.month == 6) & (t.day == 6) & (t.hour > 6))
        | (
            (t.year == 2016)
            & (t.month == 6)
            & (t.day == 6)
            & (t.hour == 6)
            & (t.minute >= 5)
        )
    ) & (
        (t.year < 2016)
        | ((t.year == 2016) & (t.month < 6))
        | ((t.year == 2016) & (t.month == 6) & (t.day < 6))
        | ((t.year == 2016) & (t.month == 6) & (t.day == 6) & (t.hour < 6))
        | (
            (t.year == 2016)
            & (t.month == 6)
            & (t.day == 6)
            & (t.hour == 6)
            & (t.minute <= 5)
        )
    )


@pytest.fixture
def base(t):
    return make_base(t)


def make_large_expr(t, base):
    src_table = t[base]
    src_table = src_table.mutate(
        _timestamp=(src_table['_timestamp'] - src_table['_timestamp'] % 3600)
        .cast('int32')
        .name('_timestamp'),
        valid_seconds=300,
    )

    aggs = []
    for meas in ['meas1', 'meas2']:
        aggs.append(src_table[meas].sum().cast('float').name(meas))
    src_table = src_table.aggregate(
        aggs, by=['_timestamp', 'dim1', 'dim2', 'valid_seconds']
    )

    part_keys = ['year', 'month', 'day', 'hour', 'minute']
    ts_col = src_table['_timestamp'].cast('timestamp')
    new_cols = {}
    for part_key in part_keys:
        part_col = getattr(ts_col, part_key)()
        new_cols[part_key] = part_col
    src_table = src_table.mutate(**new_cols)
    return src_table[
        [
            '_timestamp',
            'dim1',
            'dim2',
            'meas1',
            'meas2',
            'year',
            'month',
            'day',
            'hour',
            'minute',
        ]
    ]


@pytest.fixture
def large_expr(t, base):
    return make_large_expr(t, base)


@pytest.mark.benchmark(group="construction")
@pytest.mark.parametrize(
    "construction_fn",
    [
        pytest.param(lambda *_: make_t(), id="small"),
        pytest.param(lambda t, *_: make_base(t), id="medium"),
        pytest.param(lambda t, base: make_large_expr(t, base), id="large"),
    ],
)
def test_construction(benchmark, construction_fn, t, base):
    benchmark(construction_fn, t, base)


@pytest.mark.benchmark(group="builtins")
@pytest.mark.parametrize(
    "expr_fn",
    [
        pytest.param(lambda t, _base, _large_expr: t, id="small"),
        pytest.param(lambda _t, base, _large_expr: base, id="medium"),
        pytest.param(lambda _t, _base, large_expr: large_expr, id="large"),
    ],
)
@pytest.mark.parametrize("builtin", [hash, str])
def test_builtins(benchmark, expr_fn, builtin, t, base, large_expr):
    expr = expr_fn(t, base, large_expr)
    benchmark(builtin, expr)


@pytest.mark.benchmark(group="compilation")
@pytest.mark.parametrize("module", ["impala", "sqlite"])
@pytest.mark.parametrize(
    "expr_fn",
    [
        pytest.param(lambda t, _base, _large_expr: t, id="small"),
        pytest.param(lambda _t, base, _large_expr: base, id="medium"),
        pytest.param(lambda _t, _base, large_expr: large_expr, id="large"),
    ],
)
def test_compile(benchmark, module, expr_fn, t, base, large_expr):
    try:
        mod = getattr(ibis, module)
    except AttributeError as e:
        pytest.skip(str(e))
    else:
        expr = expr_fn(t, base, large_expr)
        benchmark(mod.compile, expr)


@pytest.fixture
def pt():
    n = 60_000
    data = pd.DataFrame(
        {
            'key': np.random.choice(16000, size=n),
            'low_card_key': np.random.choice(30, size=n),
            'value': np.random.rand(n),
            'timestamps': pd.date_range(
                start='now', periods=n, freq='s'
            ).values,
            'timestamp_strings': pd.date_range(
                start='now', periods=n, freq='s'
            ).values.astype(str),
            'repeated_timestamps': pd.date_range(
                start='2018-09-01', periods=30
            ).repeat(int(n / 30)),
        }
    )

    return ibis.pandas.connect(dict(df=data)).table('df')


def high_card_group_by(t):
    return t.groupby(t.key).aggregate(avg_value=t.value.mean())


def cast_to_dates(t):
    return t.timestamps.cast(dt.date)


def cast_to_dates_from_strings(t):
    return t.timestamp_strings.cast(dt.date)


def multikey_group_by_with_mutate(t):
    return (
        t.mutate(dates=t.timestamps.cast('date'))
        .groupby(['low_card_key', 'dates'])
        .aggregate(avg_value=lambda t: t.value.mean())
    )


def simple_sort(t):
    return t.sort_by([t.key])


def simple_sort_projection(t):
    return t[['key', 'value']].sort_by(['key'])


def multikey_sort(t):
    return t.sort_by(['low_card_key', 'key'])


def multikey_sort_projection(t):
    return t[['low_card_key', 'key', 'value']].sort_by(['low_card_key', 'key'])


def low_card_rolling_window(t):
    return ibis.trailing_range_window(
        ibis.interval(days=2),
        order_by=t.repeated_timestamps,
        group_by=t.low_card_key,
    )


def low_card_grouped_rolling(t):
    return t.value.mean().over(low_card_rolling_window(t))


def high_card_rolling_window(t):
    return ibis.trailing_range_window(
        ibis.interval(days=2),
        order_by=t.repeated_timestamps,
        group_by=t.key,
    )


def high_card_grouped_rolling(t):
    return t.value.mean().over(high_card_rolling_window(t))


@udf.reduction(['double'], 'double')
def my_mean(series):
    return series.mean()


def low_card_grouped_rolling_udf_mean(t):
    return my_mean(t.value).over(low_card_rolling_window(t))


def high_card_grouped_rolling_udf_mean(t):
    return my_mean(t.value).over(high_card_rolling_window(t))


@udf.analytic(['double'], 'double')
def my_zscore(series):
    return (series - series.mean()) / series.std()


def low_card_window(t):
    return ibis.window(group_by=t.low_card_key)


def high_card_window(t):
    return ibis.window(group_by=t.key)


def low_card_window_analytics_udf(t):
    return my_zscore(t.value).over(low_card_window(t))


def high_card_window_analytics_udf(t):
    return my_zscore(t.value).over(high_card_window(t))


@udf.reduction(['double', 'double'], 'double')
def my_wm(v, w):
    return np.average(v, weights=w)


def low_card_grouped_rolling_udf_wm(t):
    return my_wm(t.value, t.value).over(low_card_rolling_window(t))


def high_card_grouped_rolling_udf_wm(t):
    return my_wm(t.value, t.value).over(low_card_rolling_window(t))


@pytest.mark.benchmark(group="execution")
@pytest.mark.parametrize(
    "expression_fn",
    [
        pytest.param(high_card_group_by, id="high_card_group_by"),
        pytest.param(cast_to_dates, id="cast_to_dates"),
        pytest.param(
            cast_to_dates_from_strings, id="cast_to_dates_from_strings"
        ),
        pytest.param(
            multikey_group_by_with_mutate, id="multikey_group_by_with_mutate"
        ),
        pytest.param(simple_sort, id="simple_sort"),
        pytest.param(simple_sort_projection, id="simple_sort_projection"),
        pytest.param(multikey_sort, id="multikey_sort"),
        pytest.param(multikey_sort_projection, id="multikey_sort_projection"),
        pytest.param(low_card_grouped_rolling, id="low_card_grouped_rolling"),
        pytest.param(
            high_card_grouped_rolling, id="high_card_grouped_rolling"
        ),
        pytest.param(
            low_card_grouped_rolling_udf_mean,
            id="low_card_grouped_rolling_udf_mean",
        ),
        pytest.param(
            high_card_grouped_rolling_udf_mean,
            id="high_card_grouped_rolling_udf_mean",
        ),
        pytest.param(
            low_card_window_analytics_udf, id="low_card_window_analytics_udf"
        ),
        pytest.param(
            high_card_window_analytics_udf, id="high_card_window_analytics_udf"
        ),
        pytest.param(
            low_card_grouped_rolling_udf_wm,
            id="low_card_grouped_rolling_udf_wm",
        ),
        pytest.param(
            high_card_grouped_rolling_udf_wm,
            id="high_card_grouped_rolling_udf_wm",
        ),
    ],
)
def test_execute(benchmark, expression_fn, pt):
    expr = expression_fn(pt)
    benchmark(expr.execute)
