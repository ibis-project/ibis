from __future__ import annotations

from typing import IO

from public import public

import ibis
import ibis.common.exceptions as com
import ibis.expr.types as ir
from ibis.util import experimental


@public
@experimental
def parse_sql(sqlstring, catalog, dialect=None):
    """Parse a SQL string into an Ibis expression.

    Parameters
    ----------
    sqlstring : str
        SQL string to parse
    catalog : dict
        A dictionary mapping table names to either schemas or ibis table expressions.
        If a schema is passed, a table expression will be created using the schema.
    dialect : str, optional
        The SQL dialect to use with sqlglot to parse the query string.

    Returns
    -------
    expr : ir.Expr
    """
    import sqlglot as sg
    import sqlglot.optimizer as sgo
    import sqlglot.planner as sgp

    from ibis.expr.sql.core import Catalog, convert

    catalog = Catalog(
        {name: ibis.table(schema, name=name) for name, schema in catalog.items()}
    )

    expr = sg.parse_one(sqlstring, dialect)
    tree = sgo.optimize(expr, catalog.to_sqlglot(), rules=sgo.RULES)
    plan = sgp.Plan(tree)

    return convert(plan.root, catalog=catalog)


@public
def show_sql(
    expr: ir.Expr,
    dialect: str | None = None,
    file: IO[str] | None = None,
) -> None:
    """Pretty-print the compiled SQL string of an expression.

    If a dialect cannot be inferred and one was not passed, duckdb
    will be used as the dialect

    Parameters
    ----------
    expr
        Ibis expression whose SQL will be printed
    dialect
        String dialect. This is typically not required, but can be useful if
        ibis cannot infer the backend dialect.
    file
        File to write output to

    Examples
    --------
    >>> import ibis
    >>> from ibis import _
    >>> t = ibis.table(dict(a="int"), name="t")
    >>> expr = t.select(c=_.a * 2)
    >>> ibis.show_sql(expr)  # duckdb dialect by default
    SELECT
      t0.a * CAST(2 AS TINYINT) AS c
    FROM t AS t0
    >>> ibis.show_sql(expr, dialect="mysql")
    SELECT
      t0.a * 2 AS c
    FROM t AS t0
    """
    print(to_sql(expr, dialect=dialect), file=file)


class SQLString:
    """Object to hold a formatted SQL string.

    Syntax highlights in Jupyter notebooks.
    """

    __slots__ = ("sql",)

    def __init__(self, sql: str) -> None:
        self.sql = sql

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(sql={self.sql!r})"

    def __str__(self) -> str:
        return self.sql

    def _repr_markdown_(self) -> str:
        return f"```sql\n{self!s}\n```"


@public
def to_sql(expr: ir.Expr, dialect: str | None = None, **kwargs) -> SQLString:
    """Return the formatted SQL string for an expression.

    Parameters
    ----------
    expr
        Ibis expression.
    dialect
        SQL dialect to use for compilation.
    kwargs
        Scalar parameters

    Returns
    -------
    str
        Formatted SQL string
    """
    # try to infer from a non-str expression or if not possible fallback to
    # the default pretty dialect for expressions
    if dialect is None:
        try:
            backend = expr._find_backend()
        except com.IbisError:
            # default to duckdb for sqlalchemy compilation because it supports
            # the widest array of ibis features for SQL backends
            backend = ibis.duckdb
            read = "duckdb"
            write = ibis.options.sql.default_dialect
        else:
            read = write = getattr(backend, "_sqlglot_dialect", backend.name)
    else:
        try:
            backend = getattr(ibis, dialect)
        except AttributeError:
            raise ValueError(f"Unknown dialect {dialect}")
        else:
            read = write = getattr(backend, "_sqlglot_dialect", dialect)

    import sqlglot as sg

    sql = backend._to_sql(expr, **kwargs)
    (pretty,) = sg.transpile(sql, read=read, write=write, pretty=True)
    return SQLString(pretty)
