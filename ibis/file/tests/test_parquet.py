import pytest
import ibis

from pandas.util import testing as tm
pa = pytest.importorskip('pyarrow')
import pyarrow.parquet as pq  # noqa: E402

from ibis.file.parquet import ParquetClient, ParquetTable  # noqa: E402
from ibis.file.client import (
    FileDatabase, execute_and_reset as execute)  # noqa: E402


@pytest.fixture
def transformed(parquet):

    closes = parquet.pq.close
    opens = parquet.pq.open

    t = opens.inner_join(closes, ['time', 'ticker'])
    t = t[opens, closes.close]
    t = t.mutate(avg=(t.open + t.close) / 2)
    t = t[['time', 'ticker', 'avg']]
    return t


def test_creation(parquet):
    # we have existing files in our dir

    d = parquet.client.root
    assert len(list(d.iterdir())) == 1

    pqd = d / 'pq'
    assert len(list(pqd.iterdir())) == 2

    assert len(pq.read_table(str(pqd / 'open.parquet'))) == 50
    assert len(pq.read_table(str(pqd / 'close.parquet'))) == 50


def test_client(tmpdir, data):

    # construct with a path to a file
    d = tmpdir / 'pq'
    d.mkdir()

    for k, v in data.items():
        f = d / "{}.parquet".format(k)
        table = pa.Table.from_pandas(v)
        pq.write_table(table, str(f))

    c = ParquetClient(tmpdir)
    assert c.list_databases() == ['pq']
    assert c.database().pq.list_tables() == ['close', 'open']


def test_navigation(parquet):

    # directory navigation
    assert isinstance(parquet, FileDatabase)
    result = dir(parquet)
    assert result == ['pq']

    d = parquet.pq
    assert isinstance(d, FileDatabase)
    result = dir(d)
    assert result == ['close', 'open']

    result = d.list_tables()
    assert result == ['close', 'open']

    opens = d.open
    assert isinstance(opens.op(), ParquetTable)

    closes = d.close
    assert isinstance(closes.op(), ParquetTable)


def test_read(parquet, data):

    closes = parquet.pq.close
    assert str(closes) is not None

    result = closes.execute()
    expected = data['close']
    tm.assert_frame_equal(result, expected)

    result = execute(closes)
    tm.assert_frame_equal(result, expected)


def test_write(transformed, tmpdir):
    t = transformed
    expected = execute(t)

    tpath = tmpdir / 'new_dir'
    tpath.mkdir()
    path = tpath / 'foo.parquet'

    assert not path.exists()
    t = transformed[['time', 'ticker', 'avg']]
    c = ibis.parquet.connect(tpath)
    c.insert('foo.parquet', t)
    execute(t)
    assert path.exists()

    # readback
    c = ParquetClient(str(tpath)).database()
    result = c.list_databases()
    assert result == []

    result = c.foo.execute()
    tm.assert_frame_equal(result, expected)
    path = tpath / 'foo.parquet'
    assert path.exists()
