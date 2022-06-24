import decimal
import importlib
import io
import operator

import numpy as np
import pandas as pd
import pytest
import toolz
from packaging.version import parse as vparse
from pytest import param

import ibis
import ibis.common.exceptions as com
import ibis.util as util
from ibis import _
from ibis import literal as L

try:
    import duckdb
except ImportError:
    duckdb = None


@pytest.mark.parametrize(
    ('expr', 'expected'),
    [
        (ibis.NA.fillna(5), 5),
        (L(5).fillna(10), 5),
        (L(5).nullif(5), None),
        (L(10).nullif(5), 10),
    ],
)
@pytest.mark.notimpl(["datafusion"])
def test_fillna_nullif(backend, con, expr, expected):
    if expected is None:
        # The exact kind of null value used differs per backend (and version).
        # Example 1: Pandas returns np.nan while BigQuery returns None.
        # Example 2: PySpark returns np.nan if pyspark==3.0.0, but returns None
        # if pyspark <=3.0.0.
        # TODO: Make this behavior consistent (#2365)
        assert pd.isna(con.execute(expr))
    else:
        assert con.execute(expr) == expected


@pytest.mark.notimpl(
    [
        "clickhouse",
        "datafusion",
        "impala",
        "mysql",
        "postgres",
        "sqlite",
    ]
)
@pytest.mark.notyet(["duckdb"], reason="non-finite value support")
def test_isna(backend, alltypes):
    table = alltypes.mutate(na_col=np.nan)
    table = table.mutate(none_col=None)
    table = table.mutate(none_col=table['none_col'].cast('float64'))
    table_pandas = table.execute()

    for col in ['na_col', 'none_col']:
        result = table[table[col].isnan()].execute().reset_index(drop=True)

        expected = table_pandas[table_pandas[col].isna()].reset_index(
            drop=True
        )
        backend.assert_frame_equal(result, expected)


@pytest.mark.notimpl(
    ["clickhouse", "datafusion", "duckdb", "impala", "mysql", "postgres"]
)
def test_fillna(backend, alltypes):
    table = alltypes.mutate(na_col=np.nan)
    table = table.mutate(none_col=None)
    table = table.mutate(none_col=table['none_col'].cast('float64'))
    table_pandas = table.execute()

    for col in ['na_col', 'none_col']:
        result = (
            table.mutate(filled=table[col].fillna(0.0))
            .execute()
            .reset_index(drop=True)
        )

        expected = table_pandas.assign(
            filled=table_pandas[col].fillna(0.0)
        ).reset_index(drop=True)

        backend.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    ('expr', 'expected'),
    [
        param(
            ibis.coalesce(5, None, 4),
            5,
            marks=pytest.mark.notimpl(["datafusion"]),
        ),
        param(
            ibis.coalesce(ibis.NA, 4, ibis.NA),
            4,
            marks=pytest.mark.notimpl(["datafusion"]),
        ),
        param(
            ibis.coalesce(ibis.NA, ibis.NA, 3.14),
            3.14,
            marks=pytest.mark.notimpl(["datafusion"]),
        ),
    ],
)
def test_coalesce(backend, con, expr, expected):
    result = con.execute(expr)

    if isinstance(result, decimal.Decimal):
        # in case of Impala the result is decimal
        # >>> decimal.Decimal('5.56') == 5.56
        # False
        assert result == decimal.Decimal(str(expected))
    else:
        assert result == pytest.approx(expected)


# TODO(dask) - identicalTo - #2553
@pytest.mark.notimpl(["clickhouse", "datafusion", "dask", "pyspark"])
def test_identical_to(backend, alltypes, con, sorted_df):
    sorted_alltypes = alltypes.sort_by('id')
    df = sorted_df
    dt = df[['tinyint_col', 'double_col']]

    ident = sorted_alltypes.tinyint_col.identical_to(
        sorted_alltypes.double_col
    )
    expr = sorted_alltypes['id', ident.name('tmp')].sort_by('id')
    result = expr.execute().tmp

    expected = (dt.tinyint_col.isnull() & dt.double_col.isnull()) | (
        dt.tinyint_col == dt.double_col
    )

    expected = backend.default_series_rename(expected)
    backend.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    ('column', 'elements'),
    [
        ('int_col', [1, 2, 3]),
        ('int_col', (1, 2, 3)),
        ('string_col', ['1', '2', '3']),
        ('string_col', ('1', '2', '3')),
        ('int_col', {1}),
        ('int_col', frozenset({1})),
    ],
)
def test_isin(backend, alltypes, sorted_df, column, elements):
    sorted_alltypes = alltypes.sort_by('id')
    expr = sorted_alltypes[
        'id', sorted_alltypes[column].isin(elements).name('tmp')
    ].sort_by('id')
    result = expr.execute().tmp

    expected = sorted_df[column].isin(elements)
    expected = backend.default_series_rename(expected)
    backend.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    ('column', 'elements'),
    [
        ('int_col', [1, 2, 3]),
        ('int_col', (1, 2, 3)),
        ('string_col', ['1', '2', '3']),
        ('string_col', ('1', '2', '3')),
        ('int_col', {1}),
        ('int_col', frozenset({1})),
    ],
)
def test_notin(backend, alltypes, sorted_df, column, elements):
    sorted_alltypes = alltypes.sort_by('id')
    expr = sorted_alltypes[
        'id', sorted_alltypes[column].notin(elements).name('tmp')
    ].sort_by('id')
    result = expr.execute().tmp

    expected = ~sorted_df[column].isin(elements)
    expected = backend.default_series_rename(expected)
    backend.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    ('predicate_fn', 'expected_fn'),
    [
        param(lambda t: t['bool_col'], lambda df: df['bool_col'], id="no_op"),
        param(
            lambda t: ~t['bool_col'], lambda df: ~df['bool_col'], id="negate"
        ),
        param(
            lambda t: t.bool_col & t.bool_col,
            lambda df: df.bool_col & df.bool_col,
            id="and",
            marks=pytest.mark.notimpl(["datafusion"]),
        ),
        param(
            lambda t: t.bool_col | t.bool_col,
            lambda df: df.bool_col | df.bool_col,
            id="or",
            marks=pytest.mark.notimpl(["datafusion"]),
        ),
        param(
            lambda t: t.bool_col ^ t.bool_col,
            lambda df: df.bool_col ^ df.bool_col,
            id="xor",
            marks=pytest.mark.notimpl(["datafusion"]),
        ),
    ],
)
def test_filter(backend, alltypes, sorted_df, predicate_fn, expected_fn):
    sorted_alltypes = alltypes.sort_by('id')
    table = sorted_alltypes[predicate_fn(sorted_alltypes)].sort_by('id')
    result = table.execute()
    expected = sorted_df[expected_fn(sorted_df)]

    backend.assert_frame_equal(result, expected)


@pytest.mark.notimpl(
    [
        "clickhouse",
        "datafusion",
        "duckdb",
        "impala",
        "mysql",
        "postgres",
        "sqlite",
    ]
)
def test_filter_with_window_op(backend, alltypes, sorted_df):
    sorted_alltypes = alltypes.sort_by('id')
    table = sorted_alltypes
    window = ibis.window(group_by=table.id)
    table = table.filter(lambda t: t['id'].mean().over(window) > 3).sort_by(
        'id'
    )
    result = table.execute()
    expected = (
        sorted_df.groupby(['id'])
        .filter(lambda t: t['id'].mean() > 3)
        .reset_index(drop=True)
    )
    backend.assert_frame_equal(result, expected)


@pytest.mark.notimpl(["datafusion"])
def test_case_where(backend, alltypes, df):
    table = alltypes
    table = table.mutate(
        new_col=(
            ibis.case()
            .when(table['int_col'] == 1, 20)
            .when(table['int_col'] == 0, 10)
            .else_(0)
            .end()
            .cast('int64')
        )
    )

    result = table.execute()

    expected = df.copy()
    mask_0 = expected['int_col'] == 1
    mask_1 = expected['int_col'] == 0

    expected['new_col'] = 0
    expected.loc[mask_0, 'new_col'] = 20
    expected.loc[mask_1, 'new_col'] = 10

    backend.assert_frame_equal(result, expected)


# TODO: some of these are notimpl (datafusion) others are probably never
@pytest.mark.notimpl(["datafusion", "mysql", "sqlite"])
@pytest.mark.xfail(
    duckdb is not None and vparse(duckdb.__version__) < vparse("0.3.3"),
    reason="<0.3.3 does not support isnan/isinf properly",
)
def test_select_filter_mutate(backend, alltypes, df):
    """Test that select, filter and mutate are executed in right order.

    Before PR #2635, try_fusion in analysis.py would fuse these operations
    together in a way that the order of the operations were wrong. (mutate
    was executed before filter).
    """
    t = alltypes

    # Prepare the float_col so that filter must execute
    # before the cast to get the correct result.
    t = t.mutate(
        float_col=ibis.case()
        .when(t['bool_col'], t['float_col'])
        .else_(np.nan)
        .end()
    )

    # Actual test
    t = t[t.columns]
    t = t[~t['float_col'].isnan()]
    t = t.mutate(float_col=t['float_col'].cast('float64'))
    result = t.execute()

    expected = df.copy()
    expected.loc[~df['bool_col'], 'float_col'] = None
    expected = expected[~expected['float_col'].isna()].reset_index(drop=True)
    expected = expected.assign(
        float_col=expected['float_col'].astype('float64')
    )

    backend.assert_series_equal(result.float_col, expected.float_col)


def test_fillna_invalid(alltypes):
    with pytest.raises(
        com.IbisTypeError, match=r"\['invalid_col'\] is not a field in.*"
    ):
        alltypes.fillna({'invalid_col': 0.0})


def test_dropna_invalid(alltypes):
    with pytest.raises(
        com.IbisTypeError, match=r"'invalid_col' is not a field in.*"
    ):
        alltypes.dropna(subset=['invalid_col'])

    with pytest.raises(ValueError, match=r".*is not in.*"):
        alltypes.dropna(how='invalid')


@pytest.mark.parametrize(
    'replacements',
    [
        0.0,
        0,
        1,
        ({'na_col': 0.0}),
        ({'na_col': 1}),
        ({'none_col': 0.0}),
        ({'none_col': 1}),
    ],
)
@pytest.mark.notimpl(
    [
        "clickhouse",
        "datafusion",
        "impala",
        "mysql",
        "postgres",
        "sqlite",
    ]
)
@pytest.mark.notyet(["duckdb"], reason="non-finite value support")
def test_fillna_table(backend, alltypes, replacements):
    table = alltypes.mutate(na_col=np.nan)
    table = table.mutate(none_col=None)
    table = table.mutate(none_col=table['none_col'].cast('float64'))
    table_pandas = table.execute()

    result = table.fillna(replacements).execute().reset_index(drop=True)
    expected = table_pandas.fillna(replacements).reset_index(drop=True)

    # check_dtype is False here because there are dtype diffs between
    # Pyspark and Pandas on Java 8 - filling the 'none_col' with an int
    # results in float in Pyspark, and int in Pandas. This diff does
    # not exist in Java 11.
    backend.assert_frame_equal(result, expected, check_dtype=False)


def test_mutate_rename(alltypes):
    table = alltypes.select(["bool_col", "string_col"])
    table = table.mutate(dupe_col=table["bool_col"])
    result = table.execute()
    # check_dtype is False here because there are dtype diffs between
    # Pyspark and Pandas on Java 8 - filling the 'none_col' with an int
    # results in float in Pyspark, and int in Pandas. This diff does
    # not exist in Java 11.
    assert list(result.columns) == ["bool_col", "string_col", "dupe_col"]


@pytest.mark.parametrize(
    ('how', 'subset'),
    [
        ('any', None),
        ('any', []),
        ('any', ['int_col', 'na_col']),
        ('all', None),
        ('all', ['int_col', 'na_col']),
        ('all', 'none_col'),
    ],
)
@pytest.mark.notimpl(
    [
        "clickhouse",
        "datafusion",
        "impala",
        "mysql",
        "postgres",
        "sqlite",
    ]
)
@pytest.mark.notyet(["duckdb"], reason="non-finite value support")
def test_dropna_table(backend, alltypes, how, subset):
    table = alltypes.mutate(na_col=np.nan)
    table = table.mutate(none_col=None)
    table = table.mutate(none_col=table['none_col'].cast('float64'))
    table_pandas = table.execute()

    result = table.dropna(subset, how).execute().reset_index(drop=True)
    subset = util.promote_list(subset) if subset else table_pandas.columns
    expected = table_pandas.dropna(how=how, subset=subset).reset_index(
        drop=True
    )

    # check_dtype is False here because there are dtype diffs between
    # Pyspark and Pandas on Java 8 - the 'bool_col' of an empty DataFrame
    # is type object in Pyspark, and type bool in Pandas. This diff does
    # not exist in Java 11.
    backend.assert_frame_equal(result, expected, check_dtype=False)


def test_select_sort_sort(alltypes):
    query = alltypes[alltypes.year, alltypes.bool_col]
    query = query.sort_by(query.year).sort_by(query.bool_col)


def test_table_info(alltypes):
    buf = io.StringIO()
    alltypes.info(buf=buf)

    info_str = buf.getvalue()
    schema = alltypes.schema()

    assert "Nulls" in info_str
    assert all(str(type) in info_str for type in schema.types)
    assert all(name in info_str for name in schema.names)


@pytest.mark.parametrize(
    ("ibis_op", "pandas_op"),
    [
        param(
            _.string_col.isin([]),
            lambda df: df.string_col.isin([]),
            id="isin",
        ),
        param(
            _.string_col.notin([]),
            lambda df: ~df.string_col.isin([]),
            id="notin",
        ),
        param(
            (_.string_col.length() * 1).isin([1]),
            lambda df: (df.string_col.str.len() * 1).isin([1]),
            id="isin_non_empty",
        ),
        param(
            (_.string_col.length() * 1).notin([1]),
            lambda df: ~(df.string_col.str.len() * 1).isin([1]),
            id="notin_non_empty",
        ),
    ],
)
def test_isin_notin(backend, alltypes, df, ibis_op, pandas_op):
    expr = alltypes[ibis_op]
    expected = df.loc[pandas_op(df)].sort_values(["id"]).reset_index(drop=True)
    result = expr.execute().sort_values(["id"]).reset_index(drop=True)
    backend.assert_frame_equal(result, expected)


@pytest.mark.notyet(
    ["dask"],
    reason="dask doesn't support Series as isin/notin argument",
    raises=NotImplementedError,
)
@pytest.mark.notimpl(["datafusion"])
@pytest.mark.parametrize(
    ("ibis_op", "pandas_op"),
    [
        param(
            _.string_col.isin(_.string_col),
            lambda df: df.string_col.isin(df.string_col),
            id="isin_col",
        ),
        param(
            (_.bigint_col + 1).isin(_.string_col.length() + 1),
            lambda df: df.bigint_col.add(1).isin(
                df.string_col.str.len().add(1)
            ),
            id="isin_expr",
        ),
        param(
            _.string_col.notin(_.string_col),
            lambda df: ~df.string_col.isin(df.string_col),
            id="notin_col",
        ),
        param(
            (_.bigint_col + 1).notin(_.string_col.length() + 1),
            lambda df: ~(df.bigint_col.add(1)).isin(
                df.string_col.str.len().add(1)
            ),
            id="notin_expr",
        ),
    ],
)
def test_isin_notin_column_expr(backend, alltypes, df, ibis_op, pandas_op):
    expr = alltypes[ibis_op].sort_by("id")
    expected = df[pandas_op(df)].sort_values(["id"]).reset_index(drop=True)
    result = expr.execute()
    backend.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    ("expr", "expected", "op"),
    [
        param(True, True, toolz.identity, id="true_noop"),
        param(False, False, toolz.identity, id="false_noop"),
        param(True, False, operator.invert, id="true_invert"),
        param(False, True, operator.invert, id="false_invert"),
        param(True, False, operator.neg, id="true_negate"),
        param(False, True, operator.neg, id="false_negate"),
    ],
)
def test_logical_negation_literal(con, expr, expected, op):
    assert con.execute(op(ibis.literal(expr))) == expected


@pytest.mark.parametrize("op", [toolz.identity, operator.invert, operator.neg])
def test_logical_negation_column(backend, alltypes, df, op):
    result = op(alltypes["bool_col"]).execute()
    expected = op(df["bool_col"])
    backend.assert_series_equal(result, expected, check_names=False)


@pytest.mark.notimpl(["datafusion"])
@pytest.mark.parametrize(
    ("dtype", "zero", "expected"),
    [("int64", 0, 1), ("float64", 0.0, 1.0)],
)
def test_zeroifnull_literals(con, dtype, zero, expected):
    assert con.execute(ibis.NA.cast(dtype).zeroifnull()) == zero
    assert (
        con.execute(ibis.literal(expected, type=dtype).zeroifnull())
        == expected
    )


DASK_WITH_FIXED_REPLACE = vparse("2022.01.1")


def skip_if_dask_replace_is_broken(backend):
    if (name := backend.name()) != "dask":
        return
    if (
        version := vparse(importlib.import_module(name).__version__)
    ) < DASK_WITH_FIXED_REPLACE:
        pytest.skip(
            f"{name}@{version} doesn't support this operation with later "
            "versions of pandas"
        )


@pytest.mark.notimpl(["datafusion"])
def test_zeroifnull_column(backend, alltypes, df):
    skip_if_dask_replace_is_broken(backend)

    expr = alltypes.int_col.nullif(1).zeroifnull()
    result = expr.execute().astype("int32")
    expected = df.int_col.replace(1, 0).rename("tmp").astype("int32")
    backend.assert_series_equal(result, expected)


@pytest.mark.notimpl(["datafusion"])
@pytest.mark.broken(["dask"], reason="dask selection with ops.Where is broken")
def test_where_select(backend, alltypes, df):
    table = alltypes
    table = table.select(
        [
            "int_col",
            (
                ibis.where(table["int_col"] == 0, 42, -1)
                .cast("int64")
                .name("where_col")
            ),
        ]
    )

    result = table.execute()

    expected = df.loc[:, ["int_col"]].copy()

    expected['where_col'] = -1
    expected.loc[expected['int_col'] == 0, 'where_col'] = 42

    backend.assert_frame_equal(result, expected)


@pytest.mark.notimpl(["datafusion"])
def test_where_column(backend, alltypes, df):
    expr = (
        ibis.where(alltypes["int_col"] == 0, 42, -1)
        .cast("int64")
        .name("where_col")
    )
    result = expr.execute()

    expected = pd.Series(
        np.where(df.int_col == 0, 42, -1),
        name="where_col",
        dtype="int64",
    )

    backend.assert_series_equal(result, expected)
