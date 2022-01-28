from pathlib import Path

import pandas as pd

import ibis
import ibis.expr.operations as ops
import ibis.expr.types as ir
from ibis.backends.tests.base import BackendTest, RoundHalfToEven


class TestConf(BackendTest, RoundHalfToEven):
    check_names = False
    additional_skipped_operations = frozenset({ops.StringSQLLike})
    supported_to_timestamp_units = BackendTest.supported_to_timestamp_units | {
        'ns'
    }
    supports_divide_by_zero = True
    returned_timestamp_unit = 'ns'

    @staticmethod
    def connect(data_directory: Path):
        return ibis.pandas.connect(
            dictionary={
                'functional_alltypes': pd.read_csv(
                    str(data_directory / 'functional_alltypes.csv'),
                    index_col=None,
                    dtype={'bool_col': bool, 'string_col': str},
                    parse_dates=['timestamp_col'],
                    encoding='utf-8',
                ),
                'batting': pd.read_csv(str(data_directory / 'batting.csv')),
                'awards_players': pd.read_csv(
                    str(data_directory / 'awards_players.csv')
                ),
            }
        )

    @property
    def functional_alltypes(self) -> ir.TableExpr:
        return self.connection.table("functional_alltypes")

    @property
    def batting(self) -> ir.TableExpr:
        return self.connection.table("batting")

    @property
    def awards_players(self) -> ir.TableExpr:
        return self.connection.table("awards_players")
