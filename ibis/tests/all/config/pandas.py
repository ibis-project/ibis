from __future__ import absolute_import

import os

import pandas as pd

import pytest

import ibis.expr.operations as ops

from ibis.tests.all.config.backendtestconfiguration import (
    BackendTestConfiguration
)


class Pandas(BackendTestConfiguration):

    check_names = False
    additional_skipped_operations = frozenset({ops.StringSQLLike})

    @classmethod
    def connect(cls, module):
        test_data_directory = os.environ.get('IBIS_TEST_DATA_DIRECTORY')
        filename = os.path.join(test_data_directory, 'functional_alltypes.csv')
        if not os.path.exists(filename):
            pytest.skip('test data set functional_alltypes not found')
        else:
            return module.connect({
                'functional_alltypes': pd.read_csv(
                    filename,
                    index_col=None,
                    dtype={'string_col': str, 'bool_col': bool}
                )
            })
