"""User-defined functions (UDFs) implementation."""

from __future__ import annotations

import abc
import collections
import enum
import functools
import inspect
import itertools
import typing
from typing import TYPE_CHECKING, Any, Callable, Optional, TypeVar, overload

from public import public

import ibis.common.exceptions as exc
import ibis.expr.datashape as ds
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.rules as rlz
from ibis import util
from ibis.common.annotations import Argument, attribute
from ibis.common.collections import FrozenDict
from ibis.common.deferred import deferrable

if TYPE_CHECKING:
    from collections.abc import Iterable, MutableMapping

    import ibis.expr.types as ir


EMPTY = inspect.Parameter.empty


_udf_name_cache: MutableMapping[type[ops.Node], Iterable[int]] = (
    collections.defaultdict(itertools.count)
)


def _make_udf_name(name: str) -> str:
    definition = next(_udf_name_cache[name])
    return f"{name}_{definition:d}"


@enum.unique
class InputType(enum.Enum):
    BUILTIN = enum.auto()
    PANDAS = enum.auto()
    PYARROW = enum.auto()
    PYTHON = enum.auto()


@public
class ScalarUDF(ops.Impure):
    @attribute
    def shape(self):
        if not (args := getattr(self, "args")):  # noqa: B009
            # if a udf builtin takes no args then the shape check will fail
            # because there are no arguments to grab the shape of. In that case
            # default to a scalar shape
            return ds.scalar
        else:
            args = args if util.is_iterable(args) else [args]
            return rlz.highest_precedence_shape(args)


@public
class AggUDF(ops.Reduction, ops.Impure):
    where: Optional[ops.Value[dt.Boolean]] = None


def _wrap(
    wrapper,
    input_type: InputType,
    fn: Callable | None = None,
    **kwargs: Any,
) -> Callable:
    """Wrap a function `fn` with `wrapper`, allowing zero arguments when used as part of a decorator."""

    def wrap(fn):
        return functools.update_wrapper(
            deferrable(wrapper(input_type, fn, **kwargs)), fn
        )

    return wrap(fn) if fn is not None else wrap


S = TypeVar("S", bound=ops.Value)
B = TypeVar("B", bound=ops.Value)


class _UDF(abc.ABC):
    __slots__ = ()

    @property
    @abc.abstractmethod
    def _base(self) -> type[B]:
        """Base class of the UDF."""

    @classmethod
    def _make_node(
        cls,
        fn: Callable,
        input_type: InputType,
        name: str | None = None,
        database: str | None = None,
        catalog: str | None = None,
        signature: tuple[tuple, Any] | None = None,
        **kwargs,
    ) -> type[S]:
        """Construct a scalar user-defined function that is built-in to the backend."""
        if "schema" in kwargs:
            raise exc.UnsupportedArgumentError(
                """schema` is not a valid argument.
                You can use the `catalog` and `database` keywords to specify a UDF location."""
            )

        if signature is None:
            annotations = typing.get_type_hints(fn)
            if (return_annotation := annotations.pop("return", None)) is None:
                raise exc.MissingReturnAnnotationError(fn)
            fields = {
                arg_name: Argument(
                    pattern=rlz.ValueOf(annotations.get(arg_name)),
                    default=param.default,
                    typehint=annotations.get(arg_name, Any),
                )
                for arg_name, param in inspect.signature(fn).parameters.items()
            }

        else:
            arg_types, return_annotation = signature
            arg_names = list(inspect.signature(fn).parameters)
            fields = {
                arg_name: Argument(pattern=rlz.ValueOf(typ), typehint=typ)
                for arg_name, typ in zip(arg_names, arg_types)
            }

        func_name = name if name is not None else fn.__name__

        fields.update(
            {
                "dtype": dt.dtype(return_annotation),
                "__input_type__": input_type,
                # must wrap `fn` in a `property` otherwise `fn` is assumed to be a
                # method
                "__func__": property(fget=lambda _, fn=fn: fn),
                "__config__": FrozenDict(kwargs),
                "__udf_namespace__": ops.Namespace(database=database, catalog=catalog),
                "__module__": fn.__module__,
                "__func_name__": func_name,
            }
        )

        return type(_make_udf_name(fn.__name__), (cls._base,), fields)

    @classmethod
    def _make_wrapper(
        cls, input_type: InputType, fn: Callable, **kwargs: Any
    ) -> Callable:
        node = cls._make_node(fn, input_type, **kwargs)

        @functools.wraps(fn)
        def construct(*args: Any, **kwargs: Any) -> ir.Value:
            return node(*args, **kwargs).to_expr()

        return construct


@public
class scalar(_UDF):
    """Scalar user-defined functions.

    ::: {.callout-note}
    ## The `scalar` class itself is **not** a public API, its methods are.
    :::
    """

    _base = ScalarUDF

    @overload
    @classmethod
    def builtin(cls, fn: Callable) -> Callable[..., ir.Value]: ...

    @overload
    @classmethod
    def builtin(
        cls,
        *,
        name: str | None = None,
        database: str | None = None,
        catalog: str | None = None,
        signature: tuple[tuple[Any, ...], Any] | None = None,
        **kwargs: Any,
    ) -> Callable[[Callable], Callable[..., ir.Value]]: ...

    @util.experimental
    @classmethod
    def builtin(
        cls,
        fn=None,
        *,
        name=None,
        database=None,
        catalog=None,
        signature=None,
        **kwargs,
    ):
        """Construct a scalar user-defined function that is built-in to the backend.

        Parameters
        ----------
        fn
            The function to wrap.
        name
            The name of the UDF in the backend if different from the function name.
        database
            The database in which the builtin function resides.
        catalog
            The catalog in which the builtin function resides.
        signature
            If present, a tuple of the form `((arg0type, arg1type, ...), returntype)`.
            For example, a function taking an int and a float and returning a
            string would be `((int, float), str)`. If not present, the signature
            will be derived from the type annotations of the wrapped function.

            For **builtin** UDFs, only the **return type** annotation is required.
            See [the user guide](/how-to/extending/builtin.qmd#input-types) for
            more information.
        kwargs
            Additional backend-specific configuration arguments for the UDF.

        Examples
        --------
        >>> import ibis
        >>> @ibis.udf.scalar.builtin
        ... def hamming(a: str, b: str) -> int:
        ...     '''Compute the Hamming distance between two strings.'''
        >>> expr = hamming("duck", "luck")
        >>> con = ibis.connect("duckdb://")
        >>> con.execute(expr)
        1
        """
        return _wrap(
            cls._make_wrapper,
            InputType.BUILTIN,
            fn,
            name=name,
            database=database,
            catalog=catalog,
            signature=signature,
            **kwargs,
        )

    @overload
    @classmethod
    def python(cls, fn: Callable) -> Callable[..., ir.Value]: ...

    @overload
    @classmethod
    def python(
        cls,
        *,
        name: str | None = None,
        database: str | None = None,
        catalog: str | None = None,
        signature: tuple[tuple[Any, ...], Any] | None = None,
        **kwargs: Any,
    ) -> Callable[[Callable], Callable[..., ir.Value]]: ...

    @util.experimental
    @classmethod
    def python(
        cls,
        fn=None,
        *,
        name=None,
        database=None,
        catalog=None,
        signature=None,
        **kwargs,
    ):
        """Construct a **non-vectorized** scalar user-defined function that accepts Python scalar values as inputs.

        ::: {.callout-warning collapse="true"}
        ## `python` UDFs are likely to be slow

        `python` UDFs are not vectorized: they are executed row by row with one
        Python function call per row

        This calling pattern tends to be **much** slower than
        [`pandas`](/reference/scalar-udfs.qmd#ibis.expr.operations.udf.scalar.pandas)
        or
        [`pyarrow`](/reference/scalar-udfs.qmd#ibis.expr.operations.udf.scalar.pyarrow)-based
        vectorized UDFs.
        :::

        Parameters
        ----------
        fn
            The function to wrap.
        name
            The name of the UDF in the backend if different from the function name.
        database
            The database in which to create the UDF.
        catalog
            The catalog in which to create the UDF.
        signature
            If present, a tuple of the form `((arg0type, arg1type, ...), returntype)`.
            For example, a function taking an int and a float and returning a
            string would be `((int, float), str)`. If not present, the signature
            will be derived from the type annotations of the wrapped function.
        kwargs
            Additional backend-specific configuration arguments for the UDF.

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable(dict(int_col=[1, 2, 3], str_col=["a", "b", "c"]))
        >>> t
        ┏━━━━━━━━━┳━━━━━━━━━┓
        ┃ int_col ┃ str_col ┃
        ┡━━━━━━━━━╇━━━━━━━━━┩
        │ int64   │ string  │
        ├─────────┼─────────┤
        │       1 │ a       │
        │       2 │ b       │
        │       3 │ c       │
        └─────────┴─────────┘
        >>> @ibis.udf.scalar.python
        ... def str_magic(x: str) -> str:
        ...     return f"{x}_magic"
        >>> @ibis.udf.scalar.python
        ... def add_one_py(x: int) -> int:
        ...     return x + 1
        >>> str_magic(t.str_col)
        ┏━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ str_magic_0(str_col) ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━┩
        │ string               │
        ├──────────────────────┤
        │ a_magic              │
        │ b_magic              │
        │ c_magic              │
        └──────────────────────┘
        >>> add_one_py(t.int_col)
        ┏━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ add_one_py_0(int_col) ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━━┩
        │ int64                 │
        ├───────────────────────┤
        │                     2 │
        │                     3 │
        │                     4 │
        └───────────────────────┘

        Define a UDF that adds one to an integer field that is part of a struct:

        >>> import ibis.expr.datatypes as dt
        >>> FieldType = dt.Struct({"a": "int"})
        >>> @ibis.udf.scalar.python
        ... def add_one_py_struct(x: FieldType) -> int:
        ...     return x["a"] + 1
        >>> t = ibis.memtable({"struct_col": [{"a": 1}, {"a": 2}, {"a": 3}]})
        >>> add_one_py_struct(t.struct_col).name("added_one")
        ┏━━━━━━━━━━━┓
        ┃ added_one ┃
        ┡━━━━━━━━━━━┩
        │ int64     │
        ├───────────┤
        │         2 │
        │         3 │
        │         4 │
        └───────────┘

        Similarly, you can operate on maps as well:

        >>> FieldType = dt.Map(dt.string, dt.int64)
        >>> @ibis.udf.scalar.python
        ... def add_one_py_map(x: FieldType) -> int:
        ...     return x["a"] + 1
        >>> t = ibis.memtable(
        ...     {"map_col": [{"a": 1}, {"a": 2}, {"a": 3}]},
        ...     schema={"map_col": "map<string, int>"},
        ... )
        >>> add_one_py_map(t.map_col).name("added_one")
        ┏━━━━━━━━━━━┓
        ┃ added_one ┃
        ┡━━━━━━━━━━━┩
        │ int64     │
        ├───────────┤
        │         2 │
        │         3 │
        │         4 │
        └───────────┘

        See Also
        --------
        - [`pandas`](/reference/scalar-udfs.qmd#ibis.expr.operations.udf.scalar.pandas)
        - [`pyarrow`](/reference/scalar-udfs.qmd#ibis.expr.operations.udf.scalar.pyarrow)
        """
        return _wrap(
            cls._make_wrapper,
            InputType.PYTHON,
            fn,
            name=name,
            database=database,
            catalog=catalog,
            signature=signature,
            **kwargs,
        )

    @overload
    @classmethod
    def pandas(cls, fn: Callable) -> Callable[..., ir.Value]: ...

    @overload
    @classmethod
    def pandas(
        cls,
        *,
        name: str | None = None,
        database: str | None = None,
        catalog: str | None = None,
        signature: tuple[tuple[Any, ...], Any] | None = None,
        **kwargs: Any,
    ) -> Callable[[Callable], Callable[..., ir.Value]]: ...

    @util.experimental
    @classmethod
    def pandas(
        cls,
        fn=None,
        *,
        name=None,
        database=None,
        catalog=None,
        signature=None,
        **kwargs,
    ):
        """Construct a **vectorized** scalar user-defined function that accepts pandas Series' as inputs.

        Parameters
        ----------
        fn
            The function to wrap.
        name
            The name of the UDF in the backend if different from the function name.
        database
            The database in which to create the UDF.
        catalog
            The catalog in which to create the UDF.
        signature
            If present, a tuple of the form `((arg0type, arg1type, ...), returntype)`.
            For example, a function taking an int and a float and returning a
            string would be `((int, float), str)`. If not present, the signature
            will be derived from the type annotations of the wrapped function.
        kwargs
            Additional backend-specific configuration arguments for the UDF.

        Examples
        --------
        ```python
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable(dict(int_col=[1, 2, 3], str_col=["a", "b", "c"]))
        >>> t
        ┏━━━━━━━━━┳━━━━━━━━━┓
        ┃ int_col ┃ str_col ┃
        ┡━━━━━━━━━╇━━━━━━━━━┩
        │ int64   │ string  │
        ├─────────┼─────────┤
        │       1 │ a       │
        │       2 │ b       │
        │       3 │ c       │
        └─────────┴─────────┘
        >>> @ibis.udf.scalar.pandas
        ... def str_cap(x: str) -> str:
        ...     # note usage of pandas `str` method
        ...     return x.str.capitalize()
        >>> str_cap(t.str_col)  # doctest: +SKIP
        ┏━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ string_cap_0(str_col) ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━━┩
        │ string                │
        ├───────────────────────┤
        │ A                     │
        │ B                     │
        │ C                     │
        └───────────────────────┘
        ```

        Define a UDF that adds one to an integer field that is part of a struct:

        >>> import ibis.expr.datatypes as dt
        >>> FieldType = dt.Struct({"a": "int"})
        >>> @ibis.udf.scalar.pandas
        ... def add_one_py_struct(x: FieldType) -> int:
        ...     return x["a"] + 1
        >>> t = ibis.memtable({"struct_col": [{"a": 1}, {"a": 2}, {"a": 3}]})
        >>> con = ibis.pyspark.connect()
        >>> expr = add_one_py_struct(t.struct_col).name("added_one")
        >>> con.execute(expr)
        0    2
        1    3
        2    4
        Name: added_one, dtype: int64

        Similarly, you can operate on maps as well:

        >>> FieldType = dt.Map(dt.string, dt.int64)
        >>> @ibis.udf.scalar.pandas
        ... def add_one_py_map(x: FieldType) -> int:
        ...     return x.map(lambda d: d["a"] + 1)
        >>> t = ibis.memtable(
        ...     {"map_col": [{"a": 1}, {"a": 2}, {"a": 3}]},
        ...     schema={"map_col": "map<string, int>"},
        ... )
        >>> expr = add_one_py_map(t.map_col).name("added_one")
        >>> con.execute(expr)
        0    2
        1    3
        2    4
        Name: added_one, dtype: int64

        See Also
        --------
        - [`python`](/reference/scalar-udfs.qmd#ibis.expr.operations.udf.scalar.python)
        - [`pyarrow`](/reference/scalar-udfs.qmd#ibis.expr.operations.udf.scalar.pyarrow)
        """
        return _wrap(
            cls._make_wrapper,
            InputType.PANDAS,
            fn,
            name=name,
            database=database,
            catalog=catalog,
            signature=signature,
            **kwargs,
        )

    @overload
    @classmethod
    def pyarrow(cls, fn: Callable) -> Callable[..., ir.Value]: ...

    @overload
    @classmethod
    def pyarrow(
        cls,
        *,
        name: str | None = None,
        database: str | None = None,
        catalog: str | None = None,
        signature: tuple[tuple[Any, ...], Any] | None = None,
        **kwargs: Any,
    ) -> Callable[[Callable], Callable[..., ir.Value]]: ...

    @util.experimental
    @classmethod
    def pyarrow(
        cls,
        fn=None,
        *,
        name=None,
        database=None,
        catalog=None,
        signature=None,
        **kwargs,
    ):
        """Construct a **vectorized** scalar user-defined function that accepts PyArrow Arrays as input.

        Parameters
        ----------
        fn
            The function to wrap.
        name
            The name of the UDF in the backend if different from the function name.
        database
            The database in which to create the UDF.
        catalog
            The catalog in which to create the UDF.
        signature
            If present, a tuple of the form `((arg0type, arg1type, ...), returntype)`.
            For example, a function taking an int and a float and returning a
            string would be `((int, float), str)`. If not present, the signature
            will be derived from the type annotations of the wrapped function.
        kwargs
            Additional backend-specific configuration arguments for the UDF.

        Examples
        --------
        >>> import ibis
        >>> import pyarrow.compute as pc
        >>> from datetime import date
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable(
        ...     dict(start_col=[date(2024, 4, 29)], end_col=[date(2025, 4, 29)]),
        ... )
        >>> @ibis.udf.scalar.pyarrow
        ... def weeks_between(start: date, end: date) -> int:
        ...     return pc.weeks_between(start, end)
        >>> weeks_between(t.start_col, t.end_col)
        ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ weeks_between_0(start_col, end_col) ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
        │ int64                               │
        ├─────────────────────────────────────┤
        │                                  52 │
        └─────────────────────────────────────┘

        Define a UDF that adds one to an integer field that is part of a struct:

        >>> import pyarrow.compute as pac
        >>> import ibis.expr.datatypes as dt
        >>> FieldType = dt.Struct({"a": "int"})
        >>> @ibis.udf.scalar.pyarrow
        ... def add_one_py_struct(x: FieldType) -> int:
        ...     return pac.add(x.combine_chunks().field("a"), 1)
        >>> t = ibis.memtable({"struct_col": [{"a": 1}, {"a": 2}, {"a": 3}]})
        >>> add_one_py_struct(t.struct_col).name("added_one")
        ┏━━━━━━━━━━━┓
        ┃ added_one ┃
        ┡━━━━━━━━━━━┩
        │ int64     │
        ├───────────┤
        │         2 │
        │         3 │
        │         4 │
        └───────────┘

        Similarly, you can operate on maps as well:

        >>> FieldType = dt.Map(dt.string, dt.int64)
        >>> @ibis.udf.scalar.pyarrow
        ... def add_one_py_map(x: FieldType) -> int:
        ...     return pac.add(pac.map_lookup(x, "a", occurrence="first"), 1)
        >>> t = ibis.memtable(
        ...     {"map_col": [{"a": 1}, {"a": 2}, {"a": 3}]},
        ...     schema={"map_col": "map<string, int>"},
        ... )
        >>> add_one_py_map(t.map_col).name("added_one")
        ┏━━━━━━━━━━━┓
        ┃ added_one ┃
        ┡━━━━━━━━━━━┩
        │ int64     │
        ├───────────┤
        │         2 │
        │         3 │
        │         4 │
        └───────────┘

        See Also
        --------
        - [`python`](/reference/scalar-udfs.qmd#ibis.expr.operations.udf.scalar.python)
        - [`pandas`](/reference/scalar-udfs.qmd#ibis.expr.operations.udf.scalar.pandas)
        """
        return _wrap(
            cls._make_wrapper,
            InputType.PYARROW,
            fn,
            name=name,
            database=database,
            catalog=catalog,
            signature=signature,
            **kwargs,
        )


@public
class agg(_UDF):
    """Aggregate user-defined functions.

    ::: {.callout-note}
    ## The `agg` class itself is **not** a public API, its methods are.
    :::
    """

    __slots__ = ()

    _base = AggUDF

    @overload
    @classmethod
    def builtin(cls, fn: Callable) -> Callable[..., ir.Value]: ...

    @overload
    @classmethod
    def builtin(
        cls,
        *,
        name: str | None = None,
        database: str | None = None,
        catalog: str | None = None,
        signature: tuple[tuple[Any, ...], Any] | None = None,
        **kwargs: Any,
    ) -> Callable[[Callable], Callable[..., ir.Value]]: ...

    @util.experimental
    @classmethod
    def builtin(
        cls,
        fn=None,
        *,
        name=None,
        database=None,
        catalog=None,
        signature=None,
        **kwargs,
    ):
        """Construct an aggregate user-defined function that is built-in to the backend.

        Parameters
        ----------
        fn
            The function to wrap.
        name
            The name of the UDF in the backend if different from the function name.
        database
            The database in which the builtin function resides.
        catalog
            The catalog in which the builtin function resides.
        signature
            If present, a tuple of the form `((arg0type, arg1type, ...), returntype)`.
            For example, a function taking an int and a float and returning a
            string would be `((int, float), str)`. If not present, the signature
            will be derived from the type annotations of the wrapped function.
        kwargs
            Additional backend-specific configuration arguments for the UDF.

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> @ibis.udf.agg.builtin
        ... def favg(a: float) -> float:
        ...     '''Compute the average of a column using Kahan summation.'''
        >>> t = ibis.examples.penguins.fetch()
        >>> favg(t.bill_length_mm)
        ┌──────────┐
        │ 43.92193 │
        └──────────┘
        """
        return _wrap(
            cls._make_wrapper,
            InputType.BUILTIN,
            fn,
            name=name,
            database=database,
            catalog=catalog,
            signature=signature,
            **kwargs,
        )
