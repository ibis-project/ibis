from pathlib import Path
from typing import Any

import dask.dataframe as dd
import pandas as pd
import pandas.testing as tm
import pytest

import ibis
from ibis.backends.pandas.tests.conftest import TestConf as PandasTest

NPARTITIONS = 2


@pytest.fixture(scope="module")
def npartitions():
    return NPARTITIONS


class TestConf(PandasTest):
    @staticmethod
    def connect(data_directory: Path):
        # Note - we use `dd.from_pandas(pd.read_csv(...))` instead of
        # `dd.read_csv` due to https://github.com/dask/dask/issues/6970

        return ibis.dask.connect(
            {
                'functional_alltypes': dd.from_pandas(
                    pd.read_csv(
                        str(data_directory / 'functional_alltypes.csv'),
                        index_col=None,
                        dtype={'bool_col': bool, 'string_col': str},
                        parse_dates=['timestamp_col'],
                        encoding='utf-8',
                    ),
                    npartitions=NPARTITIONS,
                ),
                'batting': dd.from_pandas(
                    pd.read_csv(str(data_directory / 'batting.csv')),
                    npartitions=NPARTITIONS,
                ),
                'awards_players': dd.from_pandas(
                    pd.read_csv(str(data_directory / 'awards_players.csv')),
                    npartitions=NPARTITIONS,
                ),
            }
        )

    @classmethod
    def assert_series_equal(
        cls, left: pd.DataFrame, right: pd.DataFrame, *args: Any, **kwargs: Any
    ) -> None:
        kwargs.setdefault('check_dtype', cls.check_dtype)
        kwargs.setdefault('check_names', cls.check_names)
        left = left.reset_index(drop=True)
        right = right.reset_index(drop=True)
        tm.assert_series_equal(left, right, *args, **kwargs)


@pytest.fixture
def dataframe(npartitions):
    return dd.from_pandas(
        pd.DataFrame(
            {
                'plain_int64': list(range(1, 4)),
                'plain_strings': list('abc'),
                'dup_strings': list('dad'),
            }
        ),
        npartitions=npartitions,
    )


@pytest.fixture
def core_client(dataframe):
    return ibis.dask.connect({'df': dataframe})


@pytest.fixture
def ibis_table(core_client):
    return core_client.table('df')
