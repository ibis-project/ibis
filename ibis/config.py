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
    time_col: str = Field(
        default="time",
        description="Name of the timestamp col for execution with a timecontext See ibis.expr.timecontext for details.",  # noqa: E501
    )


class SQL(BaseModel):
    default_limit: Optional[int] = Field(
        default=10_000,
        description="Number of rows to be retrieved for an unlimited table expression. None means no limit.",  # noqa: E501
    )


class Rich(BaseModel):
    enabled: bool = Field(
        default=True,
        description="Use `rich` for table formatting if it's installed.",
    )


class Repr(BaseModel):
    rows: int = Field(
        default=10,
        description="The number of rows to display in interactive mode.",
    )
    rich: Rich = Rich()


class Options(BaseSettings):
    interactive: bool = Field(
        default=False,
        description=(
            "Show the first 10 rows of computing an expression when in a "
            "repl. The number of rows can be adjust by setting "
            "`ibis.options.repr.rows`."
        ),
    )
    verbose: bool = False
    verbose_log: Optional[Callable[[str], None]] = None
    graphviz_repr: bool = Field(
        default=True,
        description="Render expressions as GraphViz PNGs when running in a Jupyter notebook.",  # noqa: E501
    )
    default_backend: Any = None
    context_adjustment: ContextAdjustment = ContextAdjustment()
    sql: SQL = SQL()
    repr: Repr = Repr()

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
