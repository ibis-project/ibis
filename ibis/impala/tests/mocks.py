from ibis.tests.expr.mocks import BaseMockConnection


class MockImpalaConnection(BaseMockConnection):
    @property
    def dialect(self):
        from ibis.impala.compiler import ImpalaDialect

        return ImpalaDialect

    def _build_ast(self, expr, context):
        from ibis.impala.compiler import build_ast

        return build_ast(expr, context)
