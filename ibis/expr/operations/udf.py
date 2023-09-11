from __future__ import annotations

import enum
import functools
import inspect
import typing
from typing import TYPE_CHECKING, Any, Callable, TypeVar

from public import public

import ibis.common.exceptions as exc
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.rules as rlz
from ibis import util
from ibis.common.annotations import Argument
from ibis.common.collections import FrozenDict

if TYPE_CHECKING:
    import ibis.expr.types as ir


EMPTY = inspect.Parameter.empty


@enum.unique
class InputType(enum.Enum):
    PYTHON = enum.auto()
    PANDAS = enum.auto()
    PYARROW = enum.auto()
    OPAQUE = enum.auto()


@public
class ScalarUDF(ops.Value):
    shape = rlz.shape_like("args")


def _wrap(
    wrapper,
    input_type: InputType,
    fn: Callable | None = None,
    *args: Any,
    **kwargs: Any,
) -> Callable:
    """Wrap a function `fn` with `wrapper`, allowing zero arguments when used as part of a decorator."""
    if fn is None:
        return lambda fn: functools.update_wrapper(
            wrapper(input_type, fn, *args, **kwargs), fn
        )
    return functools.update_wrapper(wrapper(input_type, fn, *args, **kwargs), fn)


S = TypeVar("S", bound=ScalarUDF)


@public
class scalar:
    """Scalar user-defined functions.

    ::: {.callout-note}
    ## The `scalar` class itself is **not** a public API, its methods are.
    :::
    """

    @util.experimental
    @staticmethod
    def python(fn: Callable | None = None, *args: Any, **kwargs: Any) -> Callable:
        """Construct a **non-vectorized** scalar user-defined function that accepts Python scalar values as inputs.

        ::: {.callout-warning collapse="true"}
        ## `python` UDFs are likely to be slow

        `python` UDFs are not vectorized: they are executed row by row with one
        Python function call per row

        This calling pattern tends to be **much** slower than
        [`pandas`](./scalar-udfs.qmd#ibis.expr.operations.scalar.pandas)
        or
        [`pyarrow`](./scalar-udfs.qmd#ibis.expr.operations.scalar.pyarrow)-based
        vectorized UDFs.
        :::

        Examples
        --------
        >>> import ibis
        >>> @ibis.udf.scalar.python
        ... def add_one(x: int) -> int:
        ...     return x + 1
        ...
        >>> expr = add_one(2)
        >>> con = ibis.connect("duckdb://")
        >>> con.execute(expr)
        3

        See Also
        --------
        - [`pandas`](./scalar-udfs.qmd#ibis.expr.operations.scalar.pandas)
        - [`pyarrow`](./scalar-udfs.qmd#ibis.expr.operations.scalar.pyarrow)
        """
        return _wrap(scalar._make_wrapper, InputType.PYTHON, fn, *args, **kwargs)

    @util.experimental
    @staticmethod
    def pandas(fn: Callable | None = None, *args: Any, **kwargs: Any) -> Callable:
        """Construct a **vectorized** scalar user-defined function that accepts pandas Series' as inputs.

        Examples
        --------
        ```python
        >>> import ibis
        >>> @ibis.udf.scalar.pandas
        ... def add_one(x: int) -> int:
        ...     return x + 1
        ...
        >>> expr = add_one(2)
        >>> con = ibis.connect(os.environ["SNOWFLAKE_URL"])  # doctest: +SKIP
        >>> con.execute(expr)  # doctest: +SKIP
        3
        ```

        See Also
        --------
        - [`python`](./scalar-udfs.qmd#ibis.expr.operations.scalar.python)
        - [`pyarrow`](./scalar-udfs.qmd#ibis.expr.operations.scalar.pyarrow)
        """
        return _wrap(scalar._make_wrapper, InputType.PANDAS, fn, *args, **kwargs)

    @util.experimental
    @staticmethod
    def pyarrow(fn: Callable | None = None, *args: Any, **kwargs: Any) -> Callable:
        """Construct a **vectorized** scalar user-defined function that accepts PyArrow Arrays as input.

        Examples
        --------
        >>> import ibis
        >>> import pyarrow.compute as pc
        >>> @ibis.udf.scalar.pyarrow
        ... def add_one(x: int) -> int:
        ...     return pc.add(x, 1)
        ...
        >>> expr = add_one(2)
        >>> con = ibis.connect("duckdb://")
        >>> con.execute(expr)
        3

        See Also
        --------
        - [`python`](./scalar-udfs.qmd#ibis.expr.operations.scalar.python)
        - [`pandas`](./scalar-udfs.qmd#ibis.expr.operations.scalar.pandas)
        """
        return _wrap(scalar._make_wrapper, InputType.PYARROW, fn, *args, **kwargs)

    @staticmethod
    def _opaque(fn: Callable | None = None, *args: Any, **kwargs: Any) -> Callable:
        """Construct a scalar user-defined function that is defined outside of Python."""
        return _wrap(scalar._make_wrapper, InputType.OPAQUE, fn, *args, **kwargs)

    @staticmethod
    def _make_node(fn: Callable, input_type: InputType, *args, **kwargs) -> type[S]:
        annotations = typing.get_type_hints(fn)
        if (return_annotation := annotations.pop("return", None)) is None:
            raise exc.MissingReturnAnnotationError(fn)

        fields = {}

        for name, param in inspect.signature(fn).parameters.items():
            if (raw_dtype := annotations.get(name)) is None:
                raise exc.MissingParameterAnnotationError(fn, name)

            arg = rlz.ValueOf(dt.dtype(raw_dtype))
            fields[name] = Argument(pattern=arg, default=param.default)

        fields["dtype"] = dt.dtype(return_annotation)
        fields["__input_type__"] = input_type
        # can't be just `fn` otherwise `fn` is assumed to be a method
        fields["__func__"] = property(fget=lambda _, fn=fn: fn)
        fields["__config__"] = FrozenDict(args=args, kwargs=FrozenDict(**kwargs))
        fields["__udf_namespace__"] = kwargs.get("schema")
        fields["__module__"] = fn.__module__

        return type(fn.__name__, (ScalarUDF,), fields)

    @staticmethod
    def _make_wrapper(
        input_type: InputType, fn: Callable, *args: Any, **kwargs: Any
    ) -> Callable:
        node = scalar._make_node(fn, input_type, *args, **kwargs)

        @functools.wraps(fn)
        def construct(*args: Any, **kwargs: Any) -> ir.Value:
            return node(*args, **kwargs).to_expr()

        return construct
