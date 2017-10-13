import pytest
import pandas as pd
import numpy as np
from ibis.file.csv import CSVClient
from ibis.file.hdf5 import HDFClient


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

    csv = tmpdir.mkdir('csv_dir')

    for k, v in data.items():
        f = csv / '{}.csv'.format(k)
        v.to_csv(str(f), index=False)

    return CSVClient(tmpdir).database()


@pytest.fixture
def csv2(tmpdir, data):

    csv2 = tmpdir.mkdir('csv_dir2')
    df = pd.merge(*data.values(), on=['time', 'ticker'])
    f = csv2 / 'df.csv'
    df.to_csv(str(f), index=False)

    return CSVClient(tmpdir).database()


@pytest.fixture
def hdf(tmpdir, data):

    hdf = tmpdir.mkdir('hdf_dir')
    f = hdf / 'prices.h5'

    for k, v in data.items():
        v.to_hdf(str(f), k, format='table', data_columns=True)

    return HDFClient(tmpdir).database()
