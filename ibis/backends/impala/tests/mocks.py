from __future__ import annotations

from ibis.backends.sql.compilers import ImpalaCompiler
from ibis.tests.expr.mocks import MockBackend


class MockImpalaConnection(MockBackend):
    compiler = ImpalaCompiler
