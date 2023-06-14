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
    output_shape = rlz.shape_like("args")


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


class ScalarUDFBuilder:
    """Construct wrappers for user-defined functions."""

    @util.experimental
    def python(self, fn: Callable | None = None, *args: Any, **kwargs: Any) -> Callable:
        """Construct a scalar user-defined function that accepts Python scalar values as inputs.

        Examples
        --------
        >>> from ibis.interactive import *
        >>> @udf.scalar.python
        ... def add_one(x: int) -> int:
        ...     return x + 1
        >>> @udf.scalar.python(schema="my_udfs")
        ... def add_one(x: int) -> int:
        ...     return x + 1
        """
        return _wrap(self._make_wrapper, InputType.PYTHON, fn, *args, **kwargs)

    @util.experimental
    def pandas(self, fn: Callable | None = None, *args: Any, **kwargs: Any) -> Callable:
        """Construct a vectorized scalar user-defined function that accepts pandas Series' as inputs.

        Examples
        --------
        >>> from ibis.interactive import *
        >>> @udf.scalar.pandas
        ... def add_one(x: int) -> int:
        ...     return x + 1
        >>> @udf.scalar.pandas(schema="my_udfs")
        ... def add_one(x: int) -> int:
        ...     return x + 1
        """
        return _wrap(self._make_wrapper, InputType.PANDAS, fn, *args, **kwargs)

    @util.experimental
    def pyarrow(
        self, fn: Callable | None = None, *args: Any, **kwargs: Any
    ) -> Callable:
        """Construct a vectorized scalar user-defined function that accepts PyArrow Arrays as input.

        Examples
        --------
        >>> from ibis.interactive import *
        >>> @udf.scalar.pyarrow
        ... def add_one(x: int) -> int:
        ...     return x + 1
        >>> @udf.scalar.pyarrow(schema="my_udfs")
        ... def add_one(x: int) -> int:
        ...     return x + 1
        """
        return _wrap(self._make_wrapper, InputType.PYARROW, fn, *args, **kwargs)

    def _opaque(
        self, fn: Callable | None = None, *args: Any, **kwargs: Any
    ) -> Callable:
        """Construct a scalar user-defined function that is defined outside of Python."""
        return _wrap(self._make_wrapper, InputType.OPAQUE, fn, *args, **kwargs)

    def make_node(
        self, fn: Callable, input_type: InputType, *args, **kwargs
    ) -> type[S]:
        annotations = typing.get_type_hints(fn)
        if (return_annotation := annotations.pop("return", None)) is None:
            raise exc.MissingReturnAnnotationError(fn)

        fields = {}

        for name, param in inspect.signature(fn).parameters.items():
            if (raw_dtype := annotations.get(name)) is None:
                raise exc.MissingParameterAnnotationError(fn, name)

            arg = rlz.ValueOf(dt.dtype(raw_dtype))
            if (default := param.default) is EMPTY:
                fields[name] = Argument.required(validator=arg)
            else:
                fields[name] = Argument.default(validator=arg, default=default)

        fields["output_dtype"] = dt.dtype(return_annotation)

        fields["__input_type__"] = input_type
        # can't be just `fn` otherwise `fn` is assumed to be a method
        fields["__func__"] = property(fget=lambda _, fn=fn: fn)
        fields["__config__"] = FrozenDict(args=args, kwargs=FrozenDict(**kwargs))
        fields["__udf_namespace__"] = kwargs.get("schema")
        fields["__module__"] = fn.__module__

        return type(fn.__name__, (ScalarUDF,), fields)

    def _make_wrapper(
        self, input_type: InputType, fn: Callable, *args: Any, **kwargs: Any
    ) -> Callable:
        node = self.make_node(fn, input_type, *args, **kwargs)

        @functools.wraps(fn)
        def construct(*args: Any, **kwargs: Any) -> ir.Value:
            return node(*args, **kwargs).to_expr()

        return construct


scalar = ScalarUDFBuilder()


public(scalar=scalar)
