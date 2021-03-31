from ibis.tests.expr.mocks import BaseMockConnection


class MockImpalaConnection(BaseMockConnection):
    @property
    def dialect(self):
        from ibis.backends.impala import Backend

        return Backend().dialect

    def _build_ast(self, expr, context):
        from ibis.backends.impala.compiler import build_ast

        return build_ast(expr, context)
