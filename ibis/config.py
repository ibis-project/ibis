import contextlib
import operator
from typing import Any, Callable, Iterator, Optional, Tuple

from pydantic import BaseModel as PydanticBaseModel
from pydantic import BaseSettings, Field

__all__ = [
    "option_context",
    "options",
]


class BaseModel(PydanticBaseModel):
    class Config:
        validate_assignment = True


class ContextAdjustment(BaseModel):
    """Options related to time context adjustment."""

    time_col: str = Field(
        default="time",
        description="Name of the timestamp col for execution with a timecontext See ibis.expr.timecontext for details.",  # noqa: E501
    )


class SQL(BaseModel):
    """SQL-related options."""

    default_limit: Optional[int] = Field(
        default=10_000,
        description=(
            "Number of rows to be retrieved for a table expression without an "
            "explicit limit. [`None`][None] means no limit."
        ),
    )


class Options(BaseSettings):
    """Ibis configuration options."""

    interactive: bool = Field(
        default=False,
        description="Show the first few rows of computing an expression when in a repl.",  # noqa: E501
    )
    verbose: bool = Field(
        default=False,
        description="Run in verbose mode if [`True`][True]",
    )
    verbose_log: Optional[Callable[[str], None]] = Field(
        default=None,
        description="A callable to use when logging.",
    )
    graphviz_repr: bool = Field(
        default=False,
        description="Render expressions as GraphViz PNGs when running in a Jupyter notebook.",  # noqa: E501
    )
    default_backend: Any = Field(
        default=None,
        description="The default backend to use for execution.",
    )
    context_adjustment: ContextAdjustment = Field(
        default=ContextAdjustment(),
        description=ContextAdjustment.__doc__,
    )
    sql: SQL = Field(default=SQL(), description=SQL.__doc__)

    clickhouse: Optional[BaseModel] = None
    dask: Optional[BaseModel] = None
    impala: Optional[BaseModel] = None
    pandas: Optional[BaseModel] = None

    class Config:
        validate_assignment = True


options = Options()


def _get_namespace(key: str) -> Tuple[Any, str]:
    *prefix, field = key.split(".")
    if prefix:
        namespace = operator.attrgetter(".".join(prefix))(options)
    else:
        namespace = options
    return namespace, field


@contextlib.contextmanager
def option_context(key: str, new_value: Any) -> Iterator[None]:
    """Set the option named `key` to `new_value` inside a context manager.

    Parameters
    ----------
    key
        The dotted option key
    new_value
        The value to set `key` to
    """
    namespace, field = _get_namespace(key)
    value = getattr(namespace, field)
    setattr(namespace, field, new_value)
    try:
        yield
    finally:
        setattr(namespace, field, value)


def get_option(key: str) -> Any:
    from ibis import util

    util.warn_deprecated(
        "get_option",
        instead="get options directly: ibis.config.options.foo_option",  # noqa: E501
        version="4.0",
    )
    namespace, field = _get_namespace(key)
    return getattr(namespace, field)


def set_option(key: str, value: Any) -> None:
    from ibis import util

    util.warn_deprecated(
        "set_option",
        instead="set options directly: ibis.config.options.foo_option = 'some_value'",  # noqa: E501
        version="4.0",
    )
    namespace, field = _get_namespace(key)
    setattr(namespace, field, value)
