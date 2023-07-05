from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

import ibis
from ibis.backends.conftest import TEST_TABLES
from ibis.backends.tests.base import BackendTest, RoundHalfToEven
from ibis.backends.tests.data import array_types, json_types, struct_types, win

if TYPE_CHECKING:
    from pathlib import Path


class TestConf(BackendTest, RoundHalfToEven):
    check_names = False
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
