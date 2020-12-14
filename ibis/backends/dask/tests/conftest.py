import dask.dataframe as dd
import pandas as pd
import pytest

from .. import connect


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
