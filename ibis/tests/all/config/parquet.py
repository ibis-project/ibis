from __future__ import absolute_import

import os
import pytest

from ibis.tests.all.config.backendtestconfiguration import (
    BackendTestConfiguration
)


class Parquet(BackendTestConfiguration):

    required_modules = 'pyarrow',
    check_names = False

    @classmethod
    def connect(cls, backend):
        if not cls.data_directory.exists():
            pytest.skip('test data directory not found')

        filename = cls.data_directory / 'functional_alltypes.parquet'
        if not filename.exists():
            pytest.skip('test data set functional_alltypes.parquet not found '
                        'in test data directory')

        return backend.connect(cls.data_directory)

