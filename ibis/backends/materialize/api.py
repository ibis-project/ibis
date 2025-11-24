"""Materialize backend API functions."""

from __future__ import annotations

import ibis
import ibis.expr.types as ir
from ibis.backends.materialize import operations as mz_ops


def mz_now() -> ir.TimestampScalar:
    """Return the logical timestamp in Materialize.

    This returns Materialize's `mz_now()` function, which provides the logical
    time at which the query was executed. This is different from `ibis.now()`
    (PostgreSQL's `now()`) which returns the system clock time.

    Key differences from `now()`:
    - Returns logical timestamp (for streaming/incremental computation)
    - Can be used in temporal filters in materialized views
    - Value represents query execution time in Materialize's consistency model

    Returns
    -------
    TimestampScalar
        An expression representing Materialize's logical timestamp

    Examples
    --------
    >>> import ibis
    >>> from ibis.backends.materialize.api import mz_now
    >>> # Get the current logical timestamp
    >>> mz_now()  # doctest: +SKIP

    Use in temporal filters (e.g., last 30 seconds of data):

    >>> events = con.table("events")  # doctest: +SKIP
    >>> # Best practice: Isolate mz_now() on one side of comparison
    >>> recent = events.filter(
    ...     mz_now() > events.event_ts + ibis.interval(seconds=30)
    ... )  # doctest: +SKIP

    Compare with regular now():

    >>> # System clock time (wall clock)
    >>> ibis.now()  # doctest: +SKIP
    >>> # Logical timestamp (streaming time)
    >>> mz_now()  # doctest: +SKIP

    See Also
    --------
    ibis.now : PostgreSQL's now() function (system clock time)

    Notes
    -----
    mz_now() is fundamental to Materialize's streaming SQL model and is used
    for temporal filters in materialized views to enable incremental computation.

    **Best Practice**: When using mz_now() in temporal filters, isolate it on one
    side of the comparison for optimal incremental computation:

    - ✅ Good: `mz_now() > created_at + INTERVAL '1 day'`
    - ❌ Bad: `mz_now() - created_at > INTERVAL '1 day'`

    This pattern enables Materialize to efficiently compute incremental updates
    without reprocessing the entire dataset.

    References
    ----------
    - Function documentation: https://materialize.com/docs/sql/functions/now_and_mz_now/
    - Idiomatic patterns: https://materialize.com/docs/transform-data/idiomatic-materialize-sql/#temporal-filters
    """
    return mz_ops.MzNow().to_expr()


def mz_top_k(
    table: ir.Table,
    k: int,
    by: list[str] | str,
    order_by: list[str] | str | list[tuple[str, bool]],
    desc: bool = True,
    group_size: int | None = None,
) -> ir.Table:
    """Get top-k rows per group using idiomatic Materialize SQL.

    Parameters
    ----------
    table : Table
        The input table
    k : int
        Number of rows to keep per group
    by : str or list of str
        Column(s) to group by (partition keys)
    order_by : str or list of str or list of (str, bool)
        Column(s) to order by within each group.
        If tuple, second element is True for DESC, False for ASC.
    desc : bool, default True
        Default sort direction when order_by is just column names
    group_size : int, optional
        Materialize-specific query hint to control memory usage.
        For k=1: Sets DISTINCT ON INPUT GROUP SIZE
        For k>1: Sets LIMIT INPUT GROUP SIZE
        Ignored for non-Materialize backends.

    Returns
    -------
    Table
        Top k rows per group

    Examples
    --------
    >>> import ibis
    >>> from ibis.backends.materialize.api import mz_top_k
    >>> con = ibis.materialize.connect(...)  # doctest: +SKIP
    >>> orders = con.table("orders")  # doctest: +SKIP
    >>>
    >>> # Top 3 items per order by subtotal
    >>> mz_top_k(orders, k=3, by="order_id", order_by="subtotal", desc=True)  # doctest: +SKIP
    >>>
    >>> # Top seller per region (k=1 uses DISTINCT ON)
    >>> sales = con.table("sales")  # doctest: +SKIP
    >>> mz_top_k(sales, k=1, by="region", order_by="total_sales")  # doctest: +SKIP
    >>>
    >>> # Multiple order-by columns with explicit direction
    >>> events = con.table("events")  # doctest: +SKIP
    >>> mz_top_k(  # doctest: +SKIP
    ...     events,
    ...     k=10,
    ...     by="user_id",
    ...     order_by=[
    ...         ("priority", True),  # DESC (high priority first)
    ...         ("timestamp", False),  # ASC (oldest first)
    ...     ],
    ... )
    >>>
    >>> # Use group_size hint to optimize memory usage
    >>> mz_top_k(  # doctest: +SKIP
    ...     orders,
    ...     k=5,
    ...     by="customer_id",
    ...     order_by="order_date",
    ...     group_size=1000,  # Hint: expect ~1000 orders per customer
    ... )

    Notes
    -----
    The `group_size` parameter helps Materialize optimize memory usage by
    providing an estimate of the expected number of rows per group. This is
    particularly useful for large datasets.

    References
    ----------
    https://materialize.com/docs/transform-data/idiomatic-materialize-sql/top-k/
    https://materialize.com/docs/transform-data/optimization/#query-hints
    """
    from ibis.backends.materialize import Backend as MaterializeBackend

    # Normalize inputs
    if isinstance(by, str):
        by = [by]

    # Normalize order_by to list of (column, desc) tuples
    if isinstance(order_by, str):
        order_by = [(order_by, desc)]
    elif isinstance(order_by, list):
        if order_by and not isinstance(order_by[0], tuple):
            order_by = [(col, desc) for col in order_by]

    backend = table._find_backend()

    if isinstance(backend, MaterializeBackend):
        if k == 1:
            return _top_k_distinct_on(table, by, order_by, group_size)
        else:
            return _top_k_lateral(table, k, by, order_by, group_size)
    else:
        return _top_k_generic(table, k, by, order_by)


def _top_k_distinct_on(table, by, order_by, group_size):
    """Use DISTINCT ON for k=1 in Materialize."""
    import sqlglot as sg

    backend = table._find_backend()
    quoted = backend.compiler.quoted
    dialect = backend.dialect

    # Safely quote table name
    table_expr = sg.table(table.get_name(), quoted=quoted)
    table_sql = table_expr.sql(dialect)

    # Safely quote column identifiers for BY clause
    by_identifiers = [sg.to_identifier(col, quoted=quoted) for col in by]
    by_cols = ", ".join(id.sql(dialect) for id in by_identifiers)

    # Safely quote ORDER BY expressions
    order_parts = []
    for col, desc in order_by:
        col_id = sg.to_identifier(col, quoted=quoted)
        direction = "DESC" if desc else "ASC"
        order_parts.append(f"{col_id.sql(dialect)} {direction}")
    order_exprs = ", ".join(order_parts)

    # Validate and build OPTIONS clause
    options_clause = ""
    if group_size is not None:
        # Validate that group_size is actually an integer
        if not isinstance(group_size, int):
            raise TypeError(
                f"group_size must be an integer, got {type(group_size).__name__}"
            )
        if group_size < 0:
            raise ValueError(f"group_size must be non-negative, got {group_size}")
        options_clause = f"\n    OPTIONS (DISTINCT ON INPUT GROUP SIZE = {group_size})"

    # Build SQL with properly quoted identifiers
    # S608 false positive: All identifiers are safely quoted via sqlglot
    sql = f"""
    SELECT DISTINCT ON({by_cols}) *
    FROM {table_sql}{options_clause}
    ORDER BY {by_cols}, {order_exprs}
    """  # noqa: S608

    return backend.sql(sql)


def _top_k_lateral(table, k, by, order_by, group_size):
    """Use LATERAL join pattern for k>1 in Materialize."""
    import sqlglot as sg

    backend = table._find_backend()
    quoted = backend.compiler.quoted
    dialect = backend.dialect

    # Validate k parameter
    if not isinstance(k, int):
        raise TypeError(f"k must be an integer, got {type(k).__name__}")
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    # Safely quote table name
    table_expr = sg.table(table.get_name(), quoted=quoted)
    table_sql = table_expr.sql(dialect)

    # Safely quote BY columns
    by_identifiers = [sg.to_identifier(col, quoted=quoted) for col in by]
    by_cols = ", ".join(id.sql(dialect) for id in by_identifiers)

    # Build grp.column references for SELECT clause
    by_cols_with_prefix = ", ".join(f"grp.{id.sql(dialect)}" for id in by_identifiers)

    # Get all columns except group by columns for the lateral select
    all_cols = list(table.columns)
    lateral_cols = [col for col in all_cols if col not in by]
    lateral_identifiers = [sg.to_identifier(col, quoted=quoted) for col in lateral_cols]
    lateral_select = ", ".join(id.sql(dialect) for id in lateral_identifiers)

    # Build WHERE clause for lateral join
    where_parts = []
    for id in by_identifiers:
        col_sql = id.sql(dialect)
        where_parts.append(f"{col_sql} = grp.{col_sql}")
    where_clause = " AND ".join(where_parts)

    # Build ORDER BY for lateral subquery
    lateral_order_parts = []
    for col, desc in order_by:
        col_id = sg.to_identifier(col, quoted=quoted)
        direction = "DESC" if desc else "ASC"
        lateral_order_parts.append(f"{col_id.sql(dialect)} {direction}")
    lateral_order = ", ".join(lateral_order_parts)

    # Build final ORDER BY (same as lateral order)
    final_order_cols = lateral_order

    # Validate and build OPTIONS clause
    options_clause = ""
    if group_size is not None:
        if not isinstance(group_size, int):
            raise TypeError(
                f"group_size must be an integer, got {type(group_size).__name__}"
            )
        if group_size < 0:
            raise ValueError(f"group_size must be non-negative, got {group_size}")
        options_clause = (
            f"\n                OPTIONS (LIMIT INPUT GROUP SIZE = {group_size})"
        )

    # Build SQL with properly quoted identifiers
    # S608 false positive: All identifiers are safely quoted via sqlglot
    sql = f"""
    SELECT {by_cols_with_prefix}, lateral_data.*
    FROM (SELECT DISTINCT {by_cols} FROM {table_sql}) grp,
         LATERAL (
             SELECT {lateral_select}
             FROM {table_sql}
             WHERE {where_clause}{options_clause}
             ORDER BY {lateral_order}
             LIMIT {k}
         ) lateral_data
    ORDER BY {by_cols}, {final_order_cols}
    """  # noqa: S608

    return backend.sql(sql)


def _top_k_generic(table, k, by, order_by):
    """Generic ROW_NUMBER() implementation for non-Materialize backends."""
    # Build window function
    order_keys = [ibis.desc(col) if desc else ibis.asc(col) for col, desc in order_by]

    return (
        table.mutate(_rn=ibis.row_number().over(group_by=by, order_by=order_keys))
        .filter(ibis._["_rn"] <= k)
        .drop("_rn")
    )
