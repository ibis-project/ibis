from __future__ import absolute_import

import ibis
import pytest

from ibis.tests.all.config.backendtestconfiguration import (
    BackendTestConfiguration
)


class CSV(BackendTestConfiguration):
    check_names = False

    @classmethod
    def connect(cls, backend):
        if not cls.data_directory.exists():
            pytest.skip('test data directory not found')

        filename = cls.data_directory / 'functional_alltypes.csv'
        if not filename.exists():
            pytest.skip('test data set functional_alltypes.csv not found '
                        'in test data directory')

        return backend.connect(cls.data_directory)

    @classmethod
    def functional_alltypes(cls, con):
        schema = ibis.schema([
            ('bool_col', 'boolean'),
            ('string_col', 'string'),
        ])
        return con.table('functional_alltypes', schema=schema)
