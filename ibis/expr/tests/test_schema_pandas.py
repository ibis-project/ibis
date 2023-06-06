import pytest

import ibis.expr.datatypes as dt
import ibis.expr.schema as sch

np = pytest.importorskip('numpy')
pd = pytest.importorskip('pandas')

import pandas.testing as tm  # noqa: E402


@pytest.fixture
def df():
    return pd.DataFrame({"A": pd.Series([1], dtype="int8"), "b": ["x"]})


def test_apply_to_column_rename(df):
    schema = sch.Schema({"a": "int8", "B": "string"})
    expected = df.rename({"A": "a", "b": "B"}, axis=1)
    with pytest.warns(FutureWarning):
        new_df = schema.apply_to(df.copy())
    tm.assert_frame_equal(new_df, expected)


def test_apply_to_column_order(df):
    schema = sch.Schema({"a": "int8", "b": "string"})
    expected = df.rename({"A": "a"}, axis=1)
    with pytest.warns(FutureWarning):
        new_df = schema.apply_to(df.copy())
    tm.assert_frame_equal(new_df, expected)


def test_schema_from_to_numpy_dtypes():
    numpy_dtypes = [
        ('a', np.dtype('int64')),
        ('b', np.dtype('str')),
        ('c', np.dtype('bool')),
    ]
    ibis_schema = sch.Schema.from_numpy(numpy_dtypes)
    assert ibis_schema == sch.Schema({'a': dt.int64, 'b': dt.string, 'c': dt.boolean})

    restored_dtypes = ibis_schema.to_numpy()
    expected_dtypes = [
        ('a', np.dtype('int64')),
        ('b', np.dtype('object')),
        ('c', np.dtype('bool')),
    ]
    assert restored_dtypes == expected_dtypes


def test_schema_from_to_pandas_dtypes():
    pandas_schema = pd.Series(
        [
            ('a', np.dtype('int64')),
            ('b', np.dtype('str')),
            ('c', pd.CategoricalDtype(['a', 'b', 'c'])),
            ('d', pd.DatetimeTZDtype(tz='US/Eastern', unit='ns')),
        ]
    )
    ibis_schema = sch.schema(pandas_schema)
    expected = sch.Schema(
        {
            'a': dt.int64,
            'b': dt.string,
            'c': dt.string,
            'd': dt.Timestamp(timezone='US/Eastern'),
        }
    )
    assert ibis_schema == expected

    restored_dtypes = ibis_schema.to_pandas()
    expected_dtypes = [
        ('a', np.dtype('int64')),
        ('b', np.dtype('object')),
        ('c', np.dtype('object')),
        ('d', pd.DatetimeTZDtype(tz='US/Eastern', unit='ns')),
    ]
    assert restored_dtypes == expected_dtypes
