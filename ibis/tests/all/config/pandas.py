from __future__ import absolute_import

import os
import pytest
import pandas as pd

import ibis.expr.operations as ops

from ibis.tests.all.config.backendtestconfiguration import (
    BackendTestConfiguration
)


class Pandas(BackendTestConfiguration):
    check_names = False
    additional_skipped_operations = frozenset({ops.StringSQLLike})

    @classmethod
    def connect(cls, backend):
        filename = cls.data_directory / 'functional_alltypes.csv'

        if not os.path.exists(filename):
            pytest.skip('test data set functional_alltypes not found')

        return backend.connect({
            'functional_alltypes': pd.read_csv(
                str(filename),
                index_col=None,
                dtype={
                    'string_col': str,
                    'bool_col': bool,
                }
            )
        })
