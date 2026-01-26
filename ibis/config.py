from __future__ import annotations

from collections.abc import Callable  # noqa: TC003
from typing import Annotated, Any, Optional

from public import public

import ibis.common.exceptions as com
from ibis.common.grounds import Annotable
from ibis.common.patterns import Between

PosInt = Annotated[int, Between(lower=0)]


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


class SQL(Config):
    """SQL-related options.

    Attributes
    ----------
    fuse_selects : bool
        Whether to fuse consecutive select queries into a single query where
        possible.
    default_limit : int | None
        Number of rows to be retrieved for a table expression without an
        explicit limit. [](`None`) means no limit.
    default_dialect : str
        Dialect to use for printing SQL when the backend cannot be determined.

    """

    fuse_selects: bool = True
    default_limit: Optional[PosInt] = None
    default_dialect: str = "duckdb"


class Interactive(Config):
    """Options controlling the interactive repr.

    Attributes
    ----------
    max_rows : int
        Maximum rows to pretty print.
    max_columns : int | None
        The maximum number of columns to pretty print. If 0 (the default), the
        number of columns will be inferred from output console size. Set to
        `None` for no limit.
    max_length : int
        Maximum length for pretty-printed arrays and maps.
    max_string : int
        Maximum length for pretty-printed strings.
    max_depth : int
        Maximum depth for nested data types.
    show_types : bool
        Show the inferred type of value expressions in the interactive repr.

    """

    max_rows: int = 10
    max_columns: Optional[int] = 0
    max_length: int = 2
    max_string: int = 80
    max_depth: int = 1
    show_types: bool = True


class Repr(Config):
    """Expression printing options.

    Attributes
    ----------
    depth : int
        The maximum number of expression nodes to print when repring.
    table_columns : int
        The number of columns to show in leaf table expressions.
    table_rows : int
        The number of rows to show for in memory tables.
    query_text_length : int
        The maximum number of characters to show in the `query` field repr of
        SQLQueryResult operations.
    show_types : bool
        Show the inferred type of value expressions in the repr.
    show_variables : bool
        Show the variables in the repr instead of generated names. This is
        an advanced option and may not work in all scenarios.
    interactive : bool
        Options controlling the interactive repr.
    """

    depth: Optional[PosInt] = None
    table_columns: Optional[PosInt] = None
    table_rows: PosInt = 10
    query_text_length: PosInt = 80
    show_types: bool = False
    show_variables: bool = False
    interactive: Interactive = Interactive()


class Options(Config):
    """Ibis configuration options.

    Attributes
    ----------
    interactive : bool
        Show the first few rows of computing an expression when in a repl.
    repr : Repr
        Options controlling expression printing.
    verbose : bool
        Run in verbose mode if [](`True`)
    verbose_log: Callable[[str], None] | None
        A callable to use when logging.
    graphviz_repr : bool
        Render expressions as GraphViz PNGs when running in a Jupyter notebook.
    default_backend : Optional[ibis.backends.BaseBackend]
        The default backend to use for execution, defaults to DuckDB if not
        set.
    sql: SQL
        SQL-related options.
    clickhouse : Config | None
        Clickhouse specific options.
    impala : Config | None
        Impala specific options.
    pandas : Config | None
        Pandas specific options.
    pyspark : Config | None
        PySpark specific options.

    """

    interactive: bool = False
    repr: Repr = Repr()
    verbose: bool = False
    verbose_log: Optional[Callable] = None
    graphviz_repr: bool = False
    default_backend: Optional[Any] = None
    sql: SQL = SQL()
    clickhouse: Optional[Config] = None
    impala: Optional[Config] = None
    pandas: Optional[Config] = None
    pyspark: Optional[Config] = None


def _default_backend() -> Any:
    if (backend := options.default_backend) is not None:
        return backend

    try:
        import duckdb as _  # noqa: F401
    except ImportError:
        raise com.IbisError(
            """\
You have used a function that relies on the default backend, but the default
backend (DuckDB) is not installed.

You may specify an alternate backend to use, e.g.

ibis.set_backend("polars")

or to install the DuckDB backend, run:

    pip install 'ibis-framework[duckdb]'

or

    conda install -c conda-forge ibis-framework

For more information on available backends, visit https://ibis-project.org/install
"""
        )

    import ibis

    options.default_backend = con = ibis.duckdb.connect(":memory:")
    return con


options = Options()


public(options=options)
