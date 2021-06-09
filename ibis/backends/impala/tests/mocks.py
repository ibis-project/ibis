from ibis.backends.impala import ImpalaCompiler
from ibis.tests.expr.mocks import BaseMockConnection


class MockImpalaConnection(BaseMockConnection):
    _compiler = ImpalaCompiler
