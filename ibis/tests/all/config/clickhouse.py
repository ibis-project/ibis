import pytest

from ibis.tests.all.config.backendtestconfiguration import BackendTestConfiguration


class Clickhouse(BackendTestConfiguration):
    @classmethod
    def connect(cls, backend):
        pytest.skip('Skipping {}'.format(backend.__name__))
