from __future__ import annotations

from public import public

from .generic import AnyColumn, AnyScalar, AnyValue


@public
class NumericValue(AnyValue):
    pass  # noqa: E701,E302


@public
class NumericScalar(AnyScalar, NumericValue):
    pass  # noqa: E701,E302


@public
class NumericColumn(AnyColumn, NumericValue):
    pass  # noqa: E701,E302


@public
class IntegerValue(NumericValue):
    def convert_base(
        self,
        from_base: IntegerValue,
        to_base: IntegerValue,
    ) -> IntegerValue:
        """Convert an integer from one base to another.

        Parameters
        ----------
        from_base
            Numeric base of `arg`
        to_base
            New base

        Returns
        -------
        IntegerValue
            Converted expression
        """
        import ibis.expr.operations as ops

        return ops.BaseConvert(self, from_base, to_base).to_expr()


@public
class IntegerScalar(NumericScalar, IntegerValue):
    pass  # noqa: E701,E302


@public
class IntegerColumn(NumericColumn, IntegerValue):
    pass  # noqa: E701,E302


@public
class FloatingValue(NumericValue):
    pass  # noqa: E701,E302


@public
class FloatingScalar(NumericScalar, FloatingValue):
    pass  # noqa: E701,E302


@public
class FloatingColumn(NumericColumn, FloatingValue):
    pass  # noqa: E701,E302


@public
class DecimalValue(NumericValue):
    pass  # noqa: E701,E302


@public
class DecimalScalar(NumericScalar, DecimalValue):
    pass  # noqa: E701,E302


@public
class DecimalColumn(NumericColumn, DecimalValue):
    pass  # noqa: E701,E302
