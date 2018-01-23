from __future__ import absolute_import

import pytest

from ibis.tests.all.config.backendtestconfiguration import (
    BackendTestConfiguration
)


class HDF5(BackendTestConfiguration):

    required_modules = 'tables',

    @classmethod
    def connect(cls, module):
        pytest.skip('Skipping backend {}'.format(module.__name__))
