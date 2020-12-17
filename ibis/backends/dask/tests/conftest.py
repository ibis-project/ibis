import dask.dataframe as dd
import pandas as pd
import pytest

from ibis.backends.pandas.tests.conftest import TestConf as PandasTest

from .. import connect


class TestConf(PandasTest):
    # clone pandas directly until the rest of the dask backend is defined
    pass


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
