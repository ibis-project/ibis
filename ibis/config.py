from __future__ import annotations

import contextlib
from typing import Any, Callable

from public import public

import ibis.common.validators as rlz
from ibis.common.grounds import Annotable


class Config(Annotable):
    def get(self, key: str) -> Any:
        value = self
        for field in key.split("."):
            value = getattr(value, field)
        return value

    def set(self, key: str, value: Any) -> None:
        *prefix, key = key.split(".")
        conf = self
        for field in prefix:
            conf = getattr(conf, field)
        setattr(conf, key, value)

    @contextlib.contextmanager
    def _with_temporary(self, options):
        try:
            old = {}
            for key, value in options.items():
                old[key] = self.get(key)
                self.set(key, value)
            yield
        finally:
            for key, value in old.items():
                self.set(key, value)

    def __call__(self, options):
        return self._with_temporary(options)


class ContextAdjustment(Config):
    """Options related to time context adjustment.

    Attributes
    ----------
    time_col : str
        Name of the timestamp column for execution with a `timecontext`. See
        `ibis/expr/timecontext.py` for details.
    """

    time_col = rlz.optional(rlz.str_, default="time")


class SQL(Config):
    """SQL-related options.

    Attributes
    ----------
    default_limit : int | None
        Number of rows to be retrieved for a table expression without an
        explicit limit. [`None`][None] means no limit.
    default_dialect : str
        Dialect to use for printing SQL when the backend cannot be determined.
    """

    default_limit = rlz.optional(rlz.int_(min=0), default=10_000)
    default_dialect = rlz.optional(rlz.str_, default="duckdb")


class Repr(Config):
    """Expression printing options.

    Attributes
    ----------
    depth : int
        The maximum number of expression nodes to print when repring.
    table_columns : int
        The number of columns to show in leaf table expressions.
    query_text_length : int
        The maximum number of characters to show in the `query` field repr of
        SQLQueryResult operations.
    show_types : bool
        Show the inferred type of value expressions in the repr.
    """

    depth = rlz.optional(rlz.int_(min=0))
    table_columns = rlz.optional(rlz.int_(min=0))
    query_text_length = rlz.optional(rlz.int_(min=0), default=80)
    show_types = rlz.optional(rlz.bool_, default=False)


config_ = rlz.instance_of(Config)


class Options(Config):
    """Ibis configuration options.

    Attributes
    ----------
    interactive : bool
        Show the first few rows of computing an expression when in a repl.
    repr : Repr
        Options controlling expression printing.
    verbose : bool
        Run in verbose mode if [`True`][True]
    verbose_log: Callable[[str], None] | None
        A callable to use when logging.
    graphviz_repr : bool
        Render expressions as GraphViz PNGs when running in a Jupyter notebook.
    default_backend : Optional[str], default None
        The default backend to use for execution, defaults to DuckDB if not
        set.
    context_adjustment : ContextAdjustment
        Options related to time context adjustment.
    sql: SQL
        SQL-related options.
    clickhouse : Config | None
        Clickhouse specific options.
    dask : Config | None
        Dask specific options.
    impala : Config | None
        Impala specific options.
    pandas : Config | None
        Pandas specific options.
    pyspark : Config | None
        PySpark specific options.
    """

    interactive = rlz.optional(rlz.bool_, default=False)
    repr = rlz.optional(rlz.instance_of(Repr), default=Repr())
    verbose = rlz.optional(rlz.bool_, default=False)
    verbose_log = rlz.optional(rlz.instance_of(Callable))
    graphviz_repr = rlz.optional(rlz.bool_, default=False)
    default_backend = rlz.optional(rlz.instance_of(object))
    context_adjustment = rlz.optional(
        rlz.instance_of(ContextAdjustment), default=ContextAdjustment()
    )
    sql = rlz.optional(rlz.instance_of(SQL), default=SQL())
    clickhouse = rlz.optional(config_)
    dask = rlz.optional(config_)
    impala = rlz.optional(config_)
    pandas = rlz.optional(config_)
    pyspark = rlz.optional(config_)


_HAS_DUCKDB = True
_DUCKDB_CON = None


def _default_backend() -> Any:
    global _HAS_DUCKDB, _DUCKDB_CON

    if not _HAS_DUCKDB:
        return None

    if _DUCKDB_CON is not None:
        return _DUCKDB_CON

    try:
        import duckdb as _  # noqa: F401
    except ImportError:
        _HAS_DUCKDB = False
        return None

    import ibis

    _DUCKDB_CON = ibis.duckdb.connect(":memory:")
    return _DUCKDB_CON


options = Options()


@public
def option_context(key, new_value):
    return options({key: new_value})


public(options=options)
