from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

from public import public

from .generic import AnyColumn, AnyScalar, AnyValue, ColumnExpr, literal
from .typing import V

if TYPE_CHECKING:
    import ibis.expr.datatypes as dt

import ibis.common.exceptions as com


@public
class ArrayValue(AnyValue):
    pass  # noqa: E701,E302


@public
class ArrayScalar(AnyScalar, ArrayValue):
    pass  # noqa: E701,E302


@public
class ArrayColumn(AnyColumn, ArrayValue):
    pass  # noqa: E701,E302


@public
def array(
    values: Iterable[V],
    type: str | dt.DataType | None = None,
) -> ArrayValue:
    """Create an array expression.

    If the input expressions are all column expressions, then the output will
    be an `ArrayColumn`. The input columns will be concatenated row-wise to
    produce each array in the output array column. Each array will have length
    _n_, where _n_ is the number of input columns. All input columns should be
    of the same datatype.

    If the input expressions are Python literals, then the output will be a
    single `ArrayScalar` of length _n_, where _n_ is the number of input
    values. This is equivalent to

    ```python
    values = [1, 2, 3]
    ibis.literal(values)
    ```

    Parameters
    ----------
    values
        An iterable of Ibis expressions or a list of Python literals
    type
        An instance of `ibis.expr.datatypes.DataType` or a string indicating
        the ibis type of `value`.

    Returns
    -------
    ArrayValue
        An array column (if the inputs are column expressions), or an array
        scalar (if the inputs are Python literals)

    Examples
    --------
    Create an array column from column expressions
    >>> import ibis
    >>> t = ibis.table([('a', 'int64'), ('b', 'int64')], name='t')
    >>> result = ibis.array([t.a, t.b])

    Create an array scalar from Python literals
    >>> import ibis
    >>> result = ibis.array([1.0, 2.0, 3.0])
    """
    import ibis.expr.operations as ops

    if all([isinstance(value, ColumnExpr) for value in values]):
        return ops.ArrayColumn(values).to_expr()
    elif any([isinstance(value, ColumnExpr) for value in values]):
        raise com.IbisTypeError(
            'To create an array column using `array`, all input values must '
            'be column expressions.'
        )
    else:
        try:
            return literal(list(values), type=type)
        except com.IbisTypeError as e:
            raise com.IbisTypeError(
                'Could not create an array scalar from the values provided '
                'to `array`. Ensure that all input values have the same '
                'Python type, or can be casted to a single Python type.'
            ) from e
