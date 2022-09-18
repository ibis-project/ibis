from public import public

from ibis.expr.types import Column, Scalar, Value


@public
class JSONValue(Value):
    def __getitem__(self, key):
        import ibis.expr.operations as ops

        return ops.JSONGetItem(self, key).to_expr()


@public
class JSONScalar(Scalar, JSONValue):
    pass  # noqa: E701,E302


@public
class JSONColumn(Column, JSONValue):
    pass  # noqa: E701,E302


@public
class JSONBValue(Value):
    pass  # noqa: E701,E302


@public
class JSONBScalar(Scalar, JSONBValue):
    pass  # noqa: E701,E302


@public
class JSONBColumn(Column, JSONBValue):
    pass  # noqa: E701,E302
