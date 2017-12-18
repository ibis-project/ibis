from __future__ import absolute_import

import pytest

from ibis.tests.all.config.backendtestconfiguration import (
    BackendTestConfiguration
)


class HDF5(BackendTestConfiguration):
    @classmethod
    def connect(cls, backend):
        pytest.skip('Skipping {}'.format(backend.__name__))
