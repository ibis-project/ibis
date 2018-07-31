import pytest
import pandas as pd
import ibis

from pandas.util import testing as tm
pytest.importorskip('tables')

from ibis.file.hdf5 import HDFClient, HDFTable  # noqa: E402
from ibis.file.client import (
    FileDatabase, execute_and_reset as execute)  # noqa: E402


@pytest.fixture
def transformed(hdf):

    closes = hdf.hdf_dir.prices.close
    opens = hdf.hdf_dir.prices.open

    t = opens.inner_join(closes, ['time', 'ticker'])
    t = t[opens, closes.close]
    t = t.mutate(avg=(t.open + t.close) / 2)
    t = t[['time', 'ticker', 'avg']]
    return t


def test_creation(hdf):
    # we have existing files in our dir
    d = hdf.client.root
    assert len(list(d.iterdir())) == 1

    hdf = d / 'hdf_dir'
    assert len(list(hdf.iterdir())) == 1

    prices = str(hdf / 'prices.h5')
    assert len(pd.read_hdf(prices, 'open')) == 50
    assert len(pd.read_hdf(prices, 'close')) == 50


def test_client(tmpdir, data):

    # construct with a path to a file
    hdf = tmpdir
    f = hdf / 'prices.h5'

    for k, v in data.items():
        v.to_hdf(str(f), k, format='table', data_columns=True)

    c = HDFClient(tmpdir)
    assert c.list_databases() == ['prices']
    assert c.database().prices.list_tables() == ['close', 'open']

    c = HDFClient(tmpdir / 'prices.h5')
    assert c.list_databases() == []
    assert c.list_tables() == ['close', 'open']


def test_navigation(hdf):

    # directory navigation
    assert isinstance(hdf, FileDatabase)
    result = dir(hdf)
    assert result == ['hdf_dir']

    hdf = hdf.hdf_dir
    assert isinstance(hdf, FileDatabase)
    result = dir(hdf)
    assert result == ['prices']

    prices = hdf.prices
    assert isinstance(prices, FileDatabase)

    result = dir(prices)
    assert result == ['close', 'open']
    result = prices.list_tables()
    assert result == ['close', 'open']

    opens = prices.open
    assert isinstance(opens.op(), HDFTable)

    closes = prices.close
    assert isinstance(closes.op(), HDFTable)


def test_read(hdf, data):

    closes = hdf.hdf_dir.prices.close
    assert str(closes) is not None

    result = closes.execute()
    expected = data['close']
    tm.assert_frame_equal(result, expected)

    result = execute(closes)
    tm.assert_frame_equal(result, expected)


def test_insert(transformed, tmpdir):

    t = transformed
    expected = execute(t)

    tpath = tmpdir / 'new_dir'
    tpath.mkdir()
    path = tpath / 'foo.h5'

    assert not path.exists()
    t = transformed[['time', 'ticker', 'avg']]
    c = ibis.hdf5.connect(tpath)
    c.insert('foo.h5', 'avg', t)
    execute(t)
    assert path.exists()

    # readback
    c = HDFClient(str(tpath)).database()
    result = c.list_databases()
    assert result == ['foo']

    result = c.foo.avg.execute()
    tm.assert_frame_equal(result, expected)
    path = tpath / 'foo.h5'
    assert path.exists()
