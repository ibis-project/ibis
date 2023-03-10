import numpy as np
import pandas as pd
import pandas.testing as tm
import pytest
from pytest import param

import ibis
import ibis.expr.datatypes as dt
from ibis.backends.pandas.client import PandasTable


@pytest.fixture
def client():
    return ibis.pandas.connect(
        {
            'df': pd.DataFrame({'a': [1, 2, 3], 'b': list('abc')}),
            'df_unknown': pd.DataFrame({'array_of_strings': [['a', 'b'], [], ['c']]}),
        }
    )


@pytest.fixture
def table(client):
    return client.table('df')


@pytest.fixture
def test_data():
    test_data = test_data = pd.DataFrame({"A": [1, 2, 3, 4, 5], "B": list("abcde")})
    return test_data


def test_connect_no_args():
    con = ibis.pandas.connect()
    assert dict(con.tables) == {}


def test_client_table(table):
    assert isinstance(table.op(), ibis.expr.operations.DatabaseTable)
    assert isinstance(table.op(), PandasTable)


def test_client_table_repr(table):
    assert 'PandasTable' in repr(table)


def test_load_data(client, test_data):
    with pytest.warns(FutureWarning):
        client.load_data('testing', test_data)
    assert 'testing' in client.list_tables()
    assert client.get_schema('testing')


def test_create_table(client, test_data):
    client.create_table('testing', obj=test_data)
    assert 'testing' in client.list_tables()
    client.create_table('testingschema', schema=client.get_schema('testing'))
    assert 'testingschema' in client.list_tables()


def test_literal(client):
    lit = ibis.literal(1)
    result = client.execute(lit)
    assert result == 1


def test_list_tables(client):
    assert client.list_tables(like='df_unknown')
    assert not client.list_tables(like='not_in_the_database')
    assert client.list_tables()


def test_drop(table):
    table = table.mutate(c=table.a)
    expr = table.drop('a')
    result = expr.execute()
    expected = table[['b', 'c']].execute()
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    'unit',
    [
        'Y',
        'M',
        'D',
        'h',
        'm',
        's',
        'ms',
        'us',
        'ns',
        param('ps', marks=pytest.mark.xfail),
        param('fs', marks=pytest.mark.xfail),
        param('as', marks=pytest.mark.xfail),
    ],
)
def test_datetime64_infer(client, unit):
    value = np.datetime64('2018-01-02', unit)
    expr = ibis.literal(value, type='timestamp')
    result = client.execute(expr)
    assert result == pd.Timestamp(value).to_pydatetime()


@pytest.mark.parametrize(
    ("ext_dtype", "expected"),
    [
        (pd.StringDtype(), dt.string),
        (pd.Int8Dtype(), dt.int8),
        (pd.Int16Dtype(), dt.int16),
        (pd.Int32Dtype(), dt.int32),
        (pd.Int64Dtype(), dt.int64),
        (pd.UInt8Dtype(), dt.uint8),
        (pd.UInt16Dtype(), dt.uint16),
        (pd.UInt32Dtype(), dt.uint32),
        (pd.UInt64Dtype(), dt.uint64),
        (pd.BooleanDtype(), dt.boolean),
    ],
    ids=str,
)
def test_infer_nullable_dtypes(ext_dtype, expected):
    assert dt.dtype(ext_dtype) == expected


@pytest.mark.parametrize(
    ("arrow_dtype", "expected"),
    [
        ("string", dt.string),
        ("int8", dt.int8),
        ("int16", dt.int16),
        ("int32", dt.int32),
        ("int64", dt.int64),
        ("uint8", dt.uint8),
        ("uint16", dt.uint16),
        ("uint32", dt.uint32),
        ("uint64", dt.uint64),
        param(
            "list<item: string>",
            dt.Array(dt.string),
            marks=pytest.mark.xfail(
                reason="list repr in dtype Series argument doesn't work",
                raises=TypeError,
            ),
            id="list_string",
        ),
    ],
    ids=str,
)
def test_infer_pandas_arrow_dtype(arrow_dtype, expected):
    pytest.importorskip("pyarrow")
    ser = pd.Series([], dtype=f"{arrow_dtype}[pyarrow]")
    dtype = ser.dtype
    assert dt.dtype(dtype) == expected


def test_infer_pandas_arrow_list_dtype():
    pa = pytest.importorskip("pyarrow")
    ser = pd.Series([], dtype=pd.ArrowDtype(pa.list_(pa.string())))
    dtype = ser.dtype
    assert dt.dtype(dtype) == dt.Array(dt.string)
