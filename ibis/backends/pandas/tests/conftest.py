from pathlib import Path

import pandas as pd

import ibis
import ibis.expr.operations as ops
import ibis.expr.types as ir
from ibis.backends.tests.base import BackendTest, RoundHalfToEven
from ibis.backends.tests.data import array_types


class TestConf(BackendTest, RoundHalfToEven):
    check_names = False
    additional_skipped_operations = frozenset({ops.StringSQLLike})
    supported_to_timestamp_units = BackendTest.supported_to_timestamp_units | {'ns'}
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
                'struct': pd.DataFrame(
                    {
                        'abc': [
                            {'a': 1.0, 'b': 'banana', 'c': 2},
                            {'a': 2.0, 'b': 'apple', 'c': 3},
                            {'a': 3.0, 'b': 'orange', 'c': 4},
                            {'a': pd.NA, 'b': 'banana', 'c': 2},
                            {'a': 2.0, 'b': pd.NA, 'c': 3},
                            pd.NA,
                            {'a': 3.0, 'b': 'orange', 'c': pd.NA},
                        ]
                    }
                ),
                'json_t': pd.DataFrame(
                    {
                        "js": [
                            '{"a": [1,2,3,4], "b": 1}',
                            '{"a":null,"b":2}',
                            '{"a":"foo", "c":null}',
                            "null",
                            "[42,47,55]",
                            "[]",
                        ]
                    }
                ),
                'array_types': array_types,
            }
        )

    @property
    def functional_alltypes(self) -> ir.Table:
        return self.connection.table("functional_alltypes")

    @property
    def batting(self) -> ir.Table:
        return self.connection.table("batting")

    @property
    def awards_players(self) -> ir.Table:
        return self.connection.table("awards_players")
