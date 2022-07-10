import contextlib
import operator
from typing import Any, Callable, Iterator, Optional, Tuple

from pydantic import BaseModel as PydanticBaseModel
from pydantic import BaseSettings, Field, validator

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
    default_dialect: str = Field(
        default="duckdb",
        description=(
            "Dialect to use for printing SQL when the backend cannot be "
            "determined."
        ),
    )


class Repr(BaseModel):
    """Options controlling expression printing."""

    depth: Optional[int] = Field(
        default=None,
        description="The maximum number of expression nodes to print when repring.",  # noqa: E501
    )
    table_columns: Optional[int] = Field(
        default=None,
        description="The number of columns to show in leaf table expressions.",
    )
    query_text_length: int = Field(
        default=80,
        description="The maximum number of characters to show in the `query` field repr of SQLQueryResult operations.",  # noqa: E501
    )
    show_types: bool = Field(
        default=False,
        description="Show the inferred type of value expressions in the repr.",
    )

    @validator("depth")
    def depth_gt_zero_or_none(cls, depth: Optional[int]) -> Optional[int]:
        if depth is not None and depth <= 0:
            raise ValueError("must be None or greater than 0")
        return depth

    @validator("table_columns")
    def table_columns_gt_zero_or_none(
        cls,
        table_columns: Optional[int],
    ) -> Optional[int]:
        if table_columns is not None and table_columns <= 0:
            raise ValueError("must be None or greater than 0")
        return table_columns

    @validator("query_text_length")
    def query_text_length_ge_zero(cls, query_text_length: int) -> int:
        if query_text_length < 0:
            raise ValueError("must be non-negative")
        return query_text_length


class Options(BaseSettings):
    """Ibis configuration options."""

    interactive: bool = Field(
        default=False,
        description="Show the first few rows of computing an expression when in a repl.",  # noqa: E501
    )
    repr: Repr = Field(default=Repr(), description=Repr.__doc__)
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
    pyspark: Optional[BaseModel] = None

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
