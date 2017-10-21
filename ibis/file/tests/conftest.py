import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def df():
    # basic time/ticker frame
    rng = pd.date_range('20170101', periods=10, freq='D')
    tickers = ['GOOGL', 'FB', 'APPL', 'NFLX', 'AMZN']
    return pd.DataFrame(
        {'time': np.repeat(rng, len(tickers)),
         'ticker': np.tile(tickers, len(rng))})


@pytest.fixture
def closes(df):
    return df.assign(close=np.random.randn(len(df)))


@pytest.fixture
def opens(df):
    return df.assign(open=np.random.randn(len(df)))


@pytest.fixture
def data(opens, closes):
    return {'open': opens, 'close': closes}


@pytest.fixture
def csv(tmpdir, data):
    from ibis.file.csv import CSVClient

    csv = tmpdir.mkdir('csv_dir')

    for k, v in data.items():
        f = csv / '{}.csv'.format(k)
        v.to_csv(str(f), index=False)

    return CSVClient(tmpdir).database()


@pytest.fixture
def csv2(tmpdir, data):
    from ibis.file.csv import CSVClient

    csv2 = tmpdir.mkdir('csv_dir2')
    df = pd.merge(*data.values(), on=['time', 'ticker'])
    f = csv2 / 'df.csv'
    df.to_csv(str(f), index=False)

    return CSVClient(tmpdir).database()


@pytest.fixture
def hdf(tmpdir, data):
    from ibis.file.hdf5 import HDFClient

    hdf = tmpdir.mkdir('hdf_dir')
    f = hdf / 'prices.h5'

    for k, v in data.items():
        v.to_hdf(str(f), k, format='table', data_columns=True)

    return HDFClient(tmpdir).database()


@pytest.fixture
def parquet(tmpdir, data):
    pa = pytest.importorskip('pyarrow')
    import pyarrow.parquet as pq  # noqa: E402
    from ibis.file.parquet import ParquetClient

    # create single files
    d = tmpdir.mkdir('pq')
    for k, v in data.items():

        f = d / "{}.parquet".format(k)
        table = pa.Table.from_pandas(v)
        pq.write_table(table, str(f))

    return ParquetClient(tmpdir).database()
