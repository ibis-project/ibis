import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def file_backends_data():
    # basic time/ticker frame
    rng = pd.date_range('20170101', periods=10, freq='D')
    tickers = ['GOOGL', 'FB', 'APPL', 'NFLX', 'AMZN']
    df = pd.DataFrame(
        {
            'time': np.repeat(rng, len(tickers)),
            'ticker': np.tile(tickers, len(rng)),
        }
    )
    opens = df.assign(open=np.random.randn(len(df)))
    closes = df.assign(close=np.random.randn(len(df)))
    return {'open': opens, 'close': closes}
