import pandas as pd
import pytest

from ibis.backends.csv import CSVClient


@pytest.fixture
def csv(tmpdir, file_backends_data):
    csv = tmpdir.mkdir('csv_dir')

    for k, v in file_backends_data.items():
        f = csv / '{}.csv'.format(k)
        v.to_csv(str(f), index=False)

    return CSVClient(tmpdir).database()


@pytest.fixture
def csv2(tmpdir, file_backends_data):
    csv2 = tmpdir.mkdir('csv_dir2')
    df = pd.merge(*file_backends_data.values(), on=['time', 'ticker'])
    f = csv2 / 'df.csv'
    df.to_csv(str(f), index=False)

    return CSVClient(tmpdir).database()
