from pathlib import Path
from typing import Any

import dask.dataframe as dd
import pandas as pd
import pandas.testing as tm
import pytest

import ibis
from ibis.backends.pandas.tests.conftest import TestConf as PandasTest

from .. import connect


class TestConf(PandasTest):
    @staticmethod
    def connect(data_directory: Path) -> ibis.client.Client:
        # Note - we use `dd.from_pandas(pd.read_csv(...))` instead of
        # `dd.read_csv` due to https://github.com/dask/dask/issues/6970

        return ibis.backends.dask.connect(
            {
                'functional_alltypes': dd.from_pandas(
                    pd.read_csv(
                        str(data_directory / 'functional_alltypes.csv'),
                        index_col=None,
                        dtype={'bool_col': bool, 'string_col': str},
                        parse_dates=['timestamp_col'],
                        encoding='utf-8',
                    ),
                    npartitions=1,
                ),
                'batting': dd.from_pandas(
                    pd.read_csv(str(data_directory / 'batting.csv')),
                    npartitions=1,
                ),
                'awards_players': dd.from_pandas(
                    pd.read_csv(str(data_directory / 'awards_players.csv')),
                    npartitions=1,
                ),
            }
        )

    # @staticmethod
    # def default_series_rename(
    #     series: pd.Series, name: str = 'tmp'
    # ) -> pd.Series:
    #     return series.compute().rename(name)

    @classmethod
    def assert_series_equal(
        cls, left, right, *args: Any, **kwargs: Any
    ) -> None:
        kwargs.setdefault('check_dtype', cls.check_dtype)
        kwargs.setdefault('check_names', cls.check_names)
        # we sometimes use pandas to build the "expected" case in tests
        right = right.compute() if isinstance(right, dd.Series) else right
        tm.assert_series_equal(left.compute(), right, *args, **kwargs)

    @classmethod
    def assert_frame_equal(
        cls, left, right, *args: Any, **kwargs: Any
    ) -> None:
        left = left.compute().reset_index(drop=True)
        right = right.compute().reset_index(drop=True)
        tm.assert_frame_equal(left, right, *args, **kwargs)


@pytest.fixture
def dataframe():
    return dd.from_pandas(
        pd.DataFrame(
            {
                'plain_int64': list(range(1, 4)),
                'plain_strings': list('abc'),
                'dup_strings': list('dad'),
            }
        ),
        npartitions=1,
    )


@pytest.fixture
def core_client(dataframe):
    return connect({'df': dataframe})


@pytest.fixture
def ibis_table(core_client):
    return core_client.table('df')
