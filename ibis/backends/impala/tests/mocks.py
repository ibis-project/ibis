from ibis.backends.impala.compiler import ImpalaCompiler
from ibis.tests.expr.mocks import MockConnection


class MockImpalaConnection(MockConnection):
    compiler = ImpalaCompiler
