from pathlib import Path

import pandas as pd
import pytest

import ibis
import ibis.expr.datatypes as dt
import ibis.expr.types as ir
from ibis.backends.pandas.tests.conftest import TestConf as PandasTest


class TestConf(PandasTest):
    check_names = False
    supports_divide_by_zero = True
    returned_timestamp_unit = 'ns'

    @staticmethod
    def connect(data_directory: Path):
        filename = data_directory / 'functional_alltypes.csv'
        if not filename.exists():
            pytest.skip(f'test data set {filename} not found')
        return ibis.csv.connect(data_directory)

    @property
    def functional_alltypes(self) -> ir.TableExpr:
        schema = ibis.schema(
            [
                ('bool_col', dt.boolean),
                ('string_col', dt.string),
                ('timestamp_col', dt.timestamp),
            ]
        )
        return self.connection.table('functional_alltypes', schema=schema)

    @property
    def batting(self) -> ir.TableExpr:
        schema = ibis.schema(
            [
                ('lgID', dt.string),
                ('G', dt.float64),
                ('AB', dt.float64),
                ('R', dt.float64),
                ('H', dt.float64),
                ('X2B', dt.float64),
                ('X3B', dt.float64),
                ('HR', dt.float64),
                ('RBI', dt.float64),
                ('SB', dt.float64),
                ('CS', dt.float64),
                ('BB', dt.float64),
                ('SO', dt.float64),
            ]
        )
        return self.connection.table('batting', schema=schema)

    @property
    def awards_players(self) -> ir.TableExpr:
        schema = ibis.schema(
            [('lgID', dt.string), ('tie', dt.string), ('notes', dt.string)]
        )
        return self.connection.table('awards_players', schema=schema)


@pytest.fixture
def csv(tmpdir, file_backends_data):
    csv = tmpdir.mkdir('csv_dir')

    for k, v in file_backends_data.items():
        f = csv / f'{k}.csv'
        v.to_csv(str(f), index=False)

    return ibis.csv.connect(tmpdir).database()


@pytest.fixture
def csv2(tmpdir, file_backends_data):
    csv2 = tmpdir.mkdir('csv_dir2')
    df = pd.merge(*file_backends_data.values(), on=['time', 'ticker'])
    f = csv2 / 'df.csv'
    df.to_csv(str(f), index=False)

    return ibis.csv.connect(tmpdir).database()
