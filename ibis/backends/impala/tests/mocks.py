from ibis.backends.impala.compiler import ImpalaCompiler
from ibis.tests.expr.mocks import BaseMockConnection


class MockImpalaConnection(BaseMockConnection):
    _compiler = ImpalaCompiler
