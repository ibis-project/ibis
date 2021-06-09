from ibis.tests.expr.mocks import BaseMockConnection


class MockImpalaConnection(BaseMockConnection):
    @property
    def dialect(self):
        from ibis.backends.impala import Backend

        return Backend().dialect
