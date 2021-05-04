from ibis.tests.expr.mocks import BaseMockConnection


class MockImpalaConnection(BaseMockConnection):
    @property
    def dialect(self):
        from ibis.backends.impala import Backend

        return Backend().dialect

    def _build_ast(self, expr, context):
        from ibis.backends.impala.compiler import build_ast

        return build_ast(expr, context)


def to_sql(expr):
    """
    This used to be in compile.py, but only used for testing.
    Moved here for now, in the future we should have a standard
    and easy way to convert expressions to sql in the backend
    public API.
    """
    from ibis.backends.impala import Backend
    from ibis.backends.impala.compiler import ImpalaQueryContext

    ctx = Backend().dialect.make_context()
    return ImpalaQueryContext()._to_sql(expr, ctx)
