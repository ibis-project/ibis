from __future__ import annotations

import contextlib
import operator
from functools import singledispatch

import sqlglot as sg
import sqlglot.expressions as sge
import sqlglot.optimizer as sgo
import sqlglot.planner as sgp
from public import public

import ibis
import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.schema as sch
import ibis.expr.types as ir
from ibis.util import experimental


class Catalog(dict[str, sch.Schema]):
    """A catalog of tables and their schemas."""

    typemap = {
        dt.Int8: "tinyint",
        dt.Int16: "smallint",
        dt.Int32: "int",
        dt.Int64: "bigint",
        dt.Float16: "halffloat",
        dt.Float32: "float",
        dt.Float64: "double",
        dt.Decimal: "decimal",
        dt.Boolean: "boolean",
        dt.JSON: "json",
        dt.Interval: "interval",
        dt.Timestamp: "datetime",
        dt.Date: "date",
        dt.Binary: "varbinary",
        dt.String: "varchar",
        dt.Array: "array",
        dt.Map: "map",
        dt.UUID: "uuid",
        dt.Struct: "struct",
    }

    def to_sqlglot_dtype(self, dtype: dt.DataType) -> str:
        if dtype.is_geospatial():
            return dtype.geotype
        else:
            default = dtype.__class__.__name__.lower()
            return self.typemap.get(type(dtype), default)

    def to_sqlglot_schema(self, schema: sch.Schema) -> dict[str, str]:
        return {name: self.to_sqlglot_dtype(dtype) for name, dtype in schema.items()}

    def to_sqlglot(self):
        return {
            name: self.to_sqlglot_schema(table.schema()) for name, table in self.items()
        }

    def overlay(self, step):
        updates = {dep.name: convert(dep, catalog=self) for dep in step.dependencies}

        # handle scan aliases: FROM foo AS bar
        source = getattr(step, "source", None)
        alias = getattr(source, "args", {}).get("alias")
        if alias is not None and (source_name := self.get(source.name)) is not None:
            self[alias.name] = source_name

        return Catalog({**self, **updates})


@singledispatch
def convert(step, catalog):
    raise TypeError(type(step))


@convert.register(sgp.Scan)
def convert_scan(scan, catalog):
    catalog = catalog.overlay(scan)

    table = catalog[scan.source.alias_or_name]

    if scan.condition:
        pred = convert(scan.condition, catalog=catalog)
        table = table.filter(pred)

    if scan.projections:
        projs = [convert(proj, catalog=catalog) for proj in scan.projections]
        table = table.select(projs)

    if isinstance(scan.limit, int):
        table = table.limit(scan.limit)

    return table


@convert.register(sgp.Sort)
def convert_sort(sort, catalog):
    catalog = catalog.overlay(sort)

    table = catalog[sort.name]

    if sort.key:
        keys = [convert(key, catalog=catalog) for key in sort.key]
        table = table.order_by(keys)

    if sort.projections:
        projs = [convert(proj, catalog=catalog) for proj in sort.projections]
        table = table.select(projs)

    return table


_join_types = {
    "": "inner",
    "LEFT": "left",
    "RIGHT": "right",
}


@convert.register(sgp.Join)
def convert_join(join, catalog):
    catalog = catalog.overlay(join)

    left_name = join.name
    left_table = catalog[left_name]

    for right_name, desc in join.joins.items():
        right_table = catalog[right_name]
        join_kind = _join_types[desc["side"]]

        if desc["join_key"]:
            predicate = None
            for left_key, right_key in zip(desc["source_key"], desc["join_key"]):
                left_key = convert(left_key, catalog=catalog)
                right_key = convert(right_key, catalog=catalog)
                if predicate is None:
                    predicate = left_key == right_key
                else:
                    predicate &= left_key == right_key
        else:
            condition = desc["condition"]
            predicate = convert(condition, catalog=catalog)

        left_table = left_table.join(right_table, predicates=predicate, how=join_kind)

    if join.condition:
        predicate = convert(join.condition, catalog=catalog)
        left_table = left_table.filter(predicate)

    catalog[left_name] = left_table

    return left_table


@convert.register(sgp.Aggregate)
def convert_aggregate(agg, catalog):
    catalog = catalog.overlay(agg)

    table = catalog[agg.source]
    if agg.aggregations:
        metrics = [convert(a, catalog=catalog) for a in agg.aggregations]
        groups = [convert(g, catalog=catalog) for k, g in agg.group.items()]
        table = table.aggregate(metrics, by=groups)

    return table


@convert.register(sge.Subquery)
def convert_subquery(subquery, catalog):
    tree = sgo.optimize(subquery.this, catalog.to_sqlglot(), rules=sgo.RULES)
    plan = sgp.Plan(tree)
    return convert(plan.root, catalog=catalog)


@convert.register(sge.Literal)
def convert_literal(literal, catalog):
    value = literal.this
    if literal.is_int:
        value = int(value)
    elif literal.is_number:
        value = float(value)
    return ibis.literal(value)


@convert.register(sge.Boolean)
def convert_boolean(boolean, catalog):
    return ibis.literal(boolean.this)


@convert.register(sge.Alias)
def convert_alias(alias, catalog):
    this = convert(alias.this, catalog=catalog)
    return this.name(alias.alias)


@convert.register(sge.Column)
def convert_column(column, catalog):
    table = catalog[column.table]
    return table[column.name]


@convert.register(sge.Ordered)
def convert_ordered(ordered, catalog):
    this = convert(ordered.this, catalog=catalog)
    desc = ordered.args["desc"]  # not exposed as an attribute
    return ibis.desc(this) if desc else ibis.asc(this)


_unary_operations = {
    sge.Paren: lambda x: x,
}


@convert.register(sge.Unary)
def convert_unary(unary, catalog):
    op = _unary_operations[type(unary)]
    this = convert(unary.this, catalog=catalog)
    return op(this)


_binary_operations = {
    sge.LT: operator.lt,
    sge.LTE: operator.le,
    sge.GT: operator.gt,
    sge.GTE: operator.ge,
    sge.EQ: operator.eq,
    sge.NEQ: operator.ne,
    sge.Add: operator.add,
    sge.Sub: operator.sub,
    sge.Mul: operator.mul,
    sge.Div: operator.truediv,
    sge.Pow: operator.pow,
    sge.And: operator.and_,
    sge.Or: operator.or_,
}


@convert.register(sge.Binary)
def convert_binary(binary, catalog):
    op = _binary_operations[type(binary)]
    this = convert(binary.this, catalog=catalog)
    expr = convert(binary.expression, catalog=catalog)

    if isinstance(binary.expression, sge.Subquery):
        # expr is a table expression
        assert len(expr.columns) == 1
        name = expr.columns[0]
        expr = expr[name]

    return op(this, expr)


_reduction_methods = {
    sge.Max: "max",
    sge.Min: "min",
    sge.Quantile: "quantile",
    sge.Sum: "sum",
    sge.Avg: "mean",
    sge.Count: "count",
}


@convert.register(sge.AggFunc)
def convert_sum(reduction, catalog):
    method = _reduction_methods[type(reduction)]
    this = convert(reduction.this, catalog=catalog)
    return getattr(this, method)()


@convert.register(sge.In)
def convert_in(in_, catalog):
    this = convert(in_.this, catalog=catalog)
    candidates = [convert(expression, catalog) for expression in in_.expressions]
    return this.isin(candidates)


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
    catalog = Catalog(
        {name: ibis.table(schema, name=name) for name, schema in catalog.items()}
    )

    expr = sg.parse_one(sqlstring, dialect)
    tree = sgo.optimize(expr, catalog.to_sqlglot(), rules=sgo.RULES)
    plan = sgp.Plan(tree)

    return convert(plan.root, catalog=catalog)


class SQLString(str):
    """Object to hold a formatted SQL string.

    Syntax highlights in Jupyter notebooks.
    """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({str(self)!r})"

    def _repr_markdown_(self) -> str:
        return f"```sql\n{self!s}\n```"

    def _repr_pretty_(self, p, cycle) -> str:
        output = str(self)
        try:
            from pygments import highlight
            from pygments.formatters import TerminalFormatter
            from pygments.lexers import SqlLexer
        except ImportError:
            pass
        else:
            with contextlib.suppress(Exception):
                output = highlight(
                    code=output,
                    lexer=SqlLexer(),
                    formatter=TerminalFormatter(),
                )

        # strip trailing newline
        p.text(output.strip())


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
            # default to duckdb for SQL compilation because it supports the
            # widest array of ibis features for SQL backends
            backend = ibis.duckdb
            read = "duckdb"
            write = ibis.options.sql.default_dialect
        else:
            read = write = backend.dialect
    else:
        try:
            backend = getattr(ibis, dialect)
        except AttributeError:
            raise ValueError(f"Unknown dialect {dialect}")
        else:
            read = write = getattr(backend, "dialect", dialect)

    sql = backend._to_sql(expr.unbind(), **kwargs)
    (pretty,) = sg.transpile(sql, read=read, write=write, pretty=True)
    return SQLString(pretty)
