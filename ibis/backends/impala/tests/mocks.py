from ibis.backends.impala.compiler import ImpalaCompiler
from ibis.tests.expr.mocks import MockBackend


class MockImpalaConnection(MockBackend):
    compiler = ImpalaCompiler
