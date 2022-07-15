from pathlib import Path

import numpy as np
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
                'array_types': pd.DataFrame(
                    [
                        (
                            [np.int64(1), 2, 3],
                            ['a', 'b', 'c'],
                            [1.0, 2.0, 3.0],
                            'a',
                            1.0,
                            [[], [np.int64(1), 2, 3], None],
                        ),
                        (
                            [4, 5],
                            ['d', 'e'],
                            [4.0, 5.0],
                            'a',
                            2.0,
                            [],
                        ),
                        (
                            [6, None],
                            ['f', None],
                            [6.0, np.nan],
                            'a',
                            3.0,
                            [None, [], None],
                        ),
                        (
                            [None, 1, None],
                            [None, 'a', None],
                            [],
                            'b',
                            4.0,
                            [[1], [2], [], [3, 4, 5]],
                        ),
                        (
                            [2, None, 3],
                            ['b', None, 'c'],
                            np.nan,
                            'b',
                            5.0,
                            None,
                        ),
                        (
                            [4, None, None, 5],
                            ['d', None, None, 'e'],
                            [4.0, np.nan, np.nan, 5.0],
                            'c',
                            6.0,
                            [[1, 2, 3]],
                        ),
                    ],
                    columns=[
                        "x",
                        "y",
                        "z",
                        "grouper",
                        "scalar_column",
                        "multi_dim",
                    ],
                ),
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
