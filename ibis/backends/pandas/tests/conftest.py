from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import ibis
import ibis.expr.operations as ops
import ibis.expr.types as ir
from ibis.backends.tests.base import BackendTest, RoundHalfToEven
from ibis.backends.tests.data import array_types, json_types, struct_types, win


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
                "functional_alltypes": pd.read_csv(
                    data_directory / "functional_alltypes.csv",
                    index_col=None,
                    dtype={
                        "bool_col": bool,
                        "string_col": str,
                        "tinyint_col": np.int8,
                        "smallint_col": np.int16,
                        "int_col": np.int32,
                        "bigint_col": np.int64,
                        "float_col": np.float32,
                        "double_col": np.float64,
                    },
                    parse_dates=["timestamp_col"],
                    encoding="utf-8",
                ),
                "batting": pd.read_csv(data_directory / "batting.csv"),
                "awards_players": pd.read_csv(data_directory / "awards_players.csv"),
                'diamonds': pd.read_csv(str(data_directory / 'diamonds.csv')),
                'struct': struct_types,
                'json_t': json_types,
                'array_types': array_types,
                'win': win,
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
