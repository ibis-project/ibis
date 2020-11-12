import pytest
from pandas.util import testing as tm

import ibis
from ibis.backends.base_file import FileDatabase
from ibis.backends.csv import CSVClient, CSVTable


@pytest.fixture
def transformed(csv):

    # we need to cast to a timestamp type
    # as we read in as strings
    closes = csv.csv_dir.close
    closes = closes.mutate(time=closes.time.cast('timestamp'))
    opens = csv.csv_dir.open
    opens = opens.mutate(time=opens.time.cast('timestamp'))

    t = opens.inner_join(closes, ['time', 'ticker'])
    t = t[opens, closes.close]
    t = t.mutate(avg=(t.open + t.close) / 2)
    t = t[['time', 'ticker', 'avg']]
    return t


def test_client(tmpdir, file_backends_data):

    # construct with a path to a file
    csv = tmpdir

    for k, v in file_backends_data.items():
        f = csv / '{}.csv'.format(k)
        v.to_csv(str(f), index=False)

    c = CSVClient(csv / 'open.csv')
    assert c.list_databases() == []
    assert c.list_tables() == ['open']

    c = CSVClient(csv / 'close.csv')
    assert c.list_databases() == []
    assert c.list_tables() == ['close']


def test_navigation(csv):

    # directory navigation
    assert isinstance(csv, FileDatabase)
    result = dir(csv)
    assert result == ['csv_dir']

    prices = csv.csv_dir
    assert isinstance(prices, FileDatabase)
    result = dir(prices)
    assert result == ['close', 'open']
    result = prices.list_tables()
    assert result == ['close', 'open']

    opens = prices.open
    assert isinstance(opens.op(), CSVTable)

    closes = prices.close
    assert isinstance(closes.op(), CSVTable)


def test_read(csv, file_backends_data):

    closes = csv.csv_dir.close
    assert str(closes) is not None

    result = closes.execute()
    expected = file_backends_data['close']

    # csv's don't preserve dtypes
    expected['time'] = expected['time'].astype(str)
    tm.assert_frame_equal(result, expected)

    result = closes.execute()
    tm.assert_frame_equal(result, expected)


def test_read_with_projection(csv2, file_backends_data):

    t = csv2.csv_dir2.df
    result = t.execute()
    assert 'close' in result.columns
    assert 'open' in result.columns

    t = t[['time', 'ticker', 'close']]
    result = t.execute()
    assert 'close' in result.columns
    assert 'open' not in result.columns


def test_insert(transformed, tmpdir):
    t = transformed

    # csv's don't preserve dtypes
    expected = t.execute()
    expected['time'] = expected['time'].astype(str)

    tpath = tmpdir / 'new_csv'
    tpath.mkdir()
    path = tpath / 'foo.csv'

    assert not path.exists()
    c = ibis.csv.connect(tpath)
    c.insert('foo.csv', t)
    assert path.exists()

    # readback
    t = CSVClient(str(tpath)).database()
    result = t.list_tables()
    assert result == ['foo']

    result = t.foo.execute()
    tm.assert_frame_equal(result, expected)
