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
    BUILTIN = enum.auto()
    PANDAS = enum.auto()
    PYARROW = enum.auto()
    PYTHON = enum.auto()


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
    def python(
        fn: Callable | None = None,
        *args: Any,
        name: str | None = None,
        schema: str | None = None,
        **kwargs: Any,
    ) -> Callable:
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

        Parameters
        ----------
        fn
            The The function to wrap.
        args
            Configuration arguments for the UDF.
        name
            The name of the UDF in the backend if different from the function name.
        schema
            The schema in which to create the UDF.
        kwargs
            Additional configuration arguments for the UDF.

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
        return _wrap(
            scalar._make_wrapper,
            InputType.PYTHON,
            fn,
            *args,
            name=name,
            schema=schema,
            **kwargs,
        )

    @util.experimental
    @staticmethod
    def pandas(
        fn: Callable | None = None,
        *args: Any,
        name: str | None = None,
        schema: str | None = None,
        **kwargs: Any,
    ) -> Callable:
        """Construct a **vectorized** scalar user-defined function that accepts pandas Series' as inputs.

        Parameters
        ----------
        fn
            The The function to wrap.
        args
            Configuration arguments for the UDF.
        name
            The name of the UDF in the backend if different from the function name.
        schema
            The schema in which to create the UDF.
        kwargs
            Additional configuration arguments for the UDF.

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
        return _wrap(
            scalar._make_wrapper,
            InputType.PANDAS,
            fn,
            *args,
            name=name,
            schema=schema,
            **kwargs,
        )

    @util.experimental
    @staticmethod
    def pyarrow(
        fn: Callable | None = None,
        *args: Any,
        name: str | None = None,
        schema: str | None = None,
        **kwargs: Any,
    ) -> Callable:
        """Construct a **vectorized** scalar user-defined function that accepts PyArrow Arrays as input.

        Parameters
        ----------
        fn
            The The function to wrap.
        args
            Configuration arguments for the UDF.
        name
            The name of the UDF in the backend if different from the function name.
        schema
            The schema in which to create the UDF.
        kwargs
            Additional configuration arguments for the UDF.

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
        return _wrap(
            scalar._make_wrapper,
            InputType.PYARROW,
            fn,
            *args,
            name=name,
            schema=schema,
            **kwargs,
        )

    @util.experimental
    @staticmethod
    def builtin(
        fn: Callable | None = None,
        *args: Any,
        name: str | None = None,
        schema: str | None = None,
        **kwargs: Any,
    ) -> Callable:
        """Construct a scalar user-defined function that is built-in to the backend.

        Parameters
        ----------
        fn
            The The function to wrap.
        args
            Configuration arguments for the UDF.
        name
            The name of the UDF in the backend if different from the function name.
        schema
            The schema in which the builtin function resides.
        kwargs
            Additional configuration arguments for the UDF.

        Examples
        --------
        >>> import ibis
        >>> @ibis.udf.scalar.builtin
        ... def hamming(a: str, b: str) -> int:
        ...     '''Compute the Hamming distance between two strings.'''
        ...
        >>> expr = hamming("duck", "luck")
        >>> con = ibis.connect("duckdb://")
        >>> con.execute(expr)
        1
        """
        return _wrap(
            scalar._make_wrapper,
            InputType.BUILTIN,
            fn,
            *args,
            name=name,
            schema=schema,
            **kwargs,
        )

    @staticmethod
    def _make_node(
        fn: Callable,
        input_type: InputType,
        *args,
        name: str | None = None,
        schema: str | None = None,
        **kwargs,
    ) -> type[S]:
        """Construct a scalar user-defined function that is built-in to the backend."""

        annotations = typing.get_type_hints(fn)
        if (return_annotation := annotations.pop("return", None)) is None:
            raise exc.MissingReturnAnnotationError(fn)

        fields = {}

        func_name = name or fn.__name__

        for arg_name, param in inspect.signature(fn).parameters.items():
            if (raw_dtype := annotations.get(arg_name)) is None:
                raise exc.MissingParameterAnnotationError(fn, arg_name)

            arg = rlz.ValueOf(dt.dtype(raw_dtype))
            fields[arg_name] = Argument(pattern=arg, default=param.default)

        fields["dtype"] = dt.dtype(return_annotation)
        fields["__input_type__"] = input_type
        # can't be just `fn` otherwise `fn` is assumed to be a method
        fields["__func__"] = property(fget=lambda _, fn=fn: fn)
        fields["__config__"] = FrozenDict(args=args, kwargs=FrozenDict(**kwargs))
        fields["__udf_namespace__"] = schema
        fields["__module__"] = fn.__module__
        fields["__func_name__"] = func_name

        return type(func_name, (ScalarUDF,), fields)

    @staticmethod
    def _make_wrapper(
        input_type: InputType, fn: Callable, *args: Any, **kwargs: Any
    ) -> Callable:
        node = scalar._make_node(fn, input_type, *args, **kwargs)

        @functools.wraps(fn)
        def construct(*args: Any, **kwargs: Any) -> ir.Value:
            return node(*args, **kwargs).to_expr()

        return construct
