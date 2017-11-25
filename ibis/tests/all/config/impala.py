import pytest

from ibis.tests.all.config.backendtestconfiguration import BackendTestConfiguration


class Impala(BackendTestConfiguration):
    @classmethod
    def connect(cls, backend):
        pytest.skip('Skipping {}'.format(backend.__name__))
