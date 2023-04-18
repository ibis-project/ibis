from __future__ import annotations

from pathlib import Path

import pandas as pd

import ibis
import ibis.expr.operations as ops
from ibis.backends.conftest import TEST_TABLES
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
                **{
                    table: pd.read_parquet(
                        data_directory / "parquet" / f"{table}.parquet"
                    )
                    for table in TEST_TABLES.keys()
                },
                'struct': struct_types,
                'json_t': json_types,
                'array_types': array_types,
                'win': win,
            }
        )
