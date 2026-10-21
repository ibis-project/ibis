from __future__ import annotations

import collections
import contextlib
import decimal
import operator
import re
from dataclasses import dataclass, field
from functools import singledispatch

import sqlglot as sg
import sqlglot.expressions as sge
import sqlglot.optimizer as sgo
from public import public
from sqlglot.optimizer.annotate_types import annotate_types
from sqlglot.optimizer.canonicalize import canonicalize
from sqlglot.optimizer.qualify import qualify

import ibis
import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.schema as sch
import ibis.expr.types as ir
from ibis.backends.sql.datatypes import SqlglotType
from ibis.common.annotations import ValidationError
from ibis.common.deferred import Deferred
from ibis.util import experimental

# The optimizer rules run before conversion. ``qualify`` resolves every column
# to the source it comes from, expands ``*`` and references to SELECT-list
# aliases, and names every output column; ``annotate_types`` and
# ``canonicalize`` make the casts SQL implies explicit (a date column compared
# to a string literal). The rewriting rules (predicate pushdown, subquery
# merging, normalization) are left out: the expression follows the query as
# written.
RULES = (qualify, annotate_types, canonicalize)


def _arg(node, *names):
    """The first of ``names`` set on ``node``.

    sqlglot 28.0 renamed the ``with`` and ``from`` arguments to ``with_`` and
    ``from_``.
    """
    for name in names:
        if (value := node.args.get(name)) is not None:
            return value
    return None


class Catalog(dict[str, ir.Table]):
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


@dataclass
class _Root:
    """State shared by every scope of a query."""

    catalog: Catalog
    dialect: str | None
    # relations scanned so far; scanning one again yields a distinct ``.view()``
    seen: set[ops.Relation] = field(default_factory=set)


@dataclass
class _Member:
    """A FROM or JOIN alias: the relation to index and its column names there."""

    table: ir.Table
    columns: dict[str, str]


@dataclass
class _Aggregation:
    """Where the expressions of an aggregated SELECT ended up."""

    # an aggregate call or GROUP BY expression -> column of the aggregate
    memo: dict[sge.Expression, str]
    # (alias, column) of a plain GROUP BY column -> column of the aggregate
    groups: dict[tuple[str, str], str]


class Scope:
    """Name resolution for one SELECT.

    Column references resolve against the relation the SELECT is building
    (``rel``), which each clause replaces as it is applied: the FROM clause's
    table or join chain, then the filtered relation, then the aggregate. A
    reference to an alias of an enclosing SELECT (a correlated subquery)
    resolves through ``parent`` to a field of that SELECT's current relation.
    """

    def __init__(self, root: _Root, parent: Scope | None = None):
        self.root = root
        self.parent = parent
        self.ctes: dict[str, ir.Table] = {}
        self.members: dict[str, _Member] = {}
        self.windows: dict[str, sge.Window] = {}
        self.rel: ir.Table | None = None
        self.agg: _Aggregation | None = None
        # the SELECT list by output name, for ORDER BY
        self.outputs: dict[str, ir.Value] | None = None

    def child(self) -> Scope:
        return Scope(self.root, parent=self)

    def cte(self, name: str) -> ir.Table | None:
        scope = self
        while scope is not None:
            if (table := scope.ctes.get(name)) is not None:
                return table
            scope = scope.parent
        return None

    def bind(self, alias: str, table: ir.Table) -> None:
        self.members[alias] = _Member(table, dict(zip(table.columns, table.columns)))

    def rebind(self, rel: ir.Table) -> None:
        """Continue building from ``rel``, which keeps the column names."""
        self.rel = rel
        for member in self.members.values():
            member.table = rel

    def column(self, alias: str, name: str) -> ir.Value:
        if not alias:
            if self.outputs is not None and name in self.outputs:
                return self.outputs[name]
            raise com.IbisInputError(f"Cannot resolve column {name!r}")
        if (member := self.members.get(alias)) is not None:
            if self.agg is not None:
                if (column := self.agg.groups.get((alias, name))) is None:
                    raise com.IbisInputError(
                        f"Column {alias}.{name} must appear in the GROUP BY clause "
                        "or be used in an aggregate function"
                    )
                return self.rel[column]
            if (column := member.columns.get(name)) is None:
                raise com.IbisInputError(f"Column {name!r} is not found in {alias!r}")
            return member.table[column]
        if self.parent is not None:
            return self.parent.column(alias, name)
        raise com.IbisInputError(f"Unknown table or alias {alias!r}")


def _unsupported(node, scope, what=None):
    dialect = scope.root.dialect if scope is not None else None
    sql = node.sql(dialect=dialect) if isinstance(node, sge.Expression) else str(node)
    if what is None:
        what = type(node).__name__
        if isinstance(node, sge.Func) and (call := re.match(r"[\w.]+(?=\()", sql)):
            what = f"The function {call.group()}"
    return com.UnsupportedOperationError(f"{what} is not supported: {sql}")


def _invalid(node, scope, why):
    """The SQL is not valid, or not translatable, for a reason Ibis found."""
    sql = node.sql(dialect=scope.root.dialect)
    return com.IbisInputError(f"{why}: {sql}")


def _boolean(node, scope):
    """Convert a condition; SQL implies a boolean where Ibis requires one."""
    value = convert(node, scope)
    if not value.type().is_boolean():
        raise _invalid(node, scope, f"The condition is {value.type()}, not boolean")
    return value


def _reject_aggregates(nodes, scope, clause, *, aggregates=True) -> None:
    """Window functions, and ``aggregates`` too, are not allowed in ``clause``."""
    for root in nodes:
        for node in root.walk(
            prune=lambda n, root=root: (
                n is not root and isinstance(n, (sge.Select, sge.Subquery))
            )
        ):
            if isinstance(node, sge.Window):
                raise _invalid(
                    node, scope, f"A window function is not allowed in {clause}"
                )
            if (
                aggregates
                and isinstance(node, sge.AggFunc)
                and not _is_window_function(node)
            ):
                raise _invalid(
                    node, scope, f"An aggregate function is not allowed in {clause}"
                )


def _is_window_function(node) -> bool:
    """Whether an aggregate call is the function of a window."""
    unit = node.parent if isinstance(node.parent, sge.Filter) else node
    return isinstance(unit.parent, sge.Window) and unit.parent.this is unit


def _unparen(node):
    while isinstance(node, sge.Paren):
        node = node.this
    return node


def _conjuncts(node):
    """The AND-ed conditions of a WHERE, HAVING or ON clause."""
    if node is None:
        return []
    if isinstance(node, (sge.Where, sge.Having)):
        node = node.this
    node = _unparen(node)
    if isinstance(node, sge.And):
        return _conjuncts(node.left) + _conjuncts(node.right)
    return [node]


def _tables(node) -> set[str]:
    return {column.table for column in node.find_all(sge.Column)}


def _int(node, scope) -> int:
    if isinstance(node, sge.Literal) and node.is_int:
        return int(node.this)
    raise _unsupported(node, scope, "A LIMIT or OFFSET that is not an integer")


def _unique_names(values):
    """SQL allows duplicate output names, Ibis does not: suffix the repeats."""
    seen = collections.Counter()
    out = []
    for value in values:
        name = value.get_name()
        if (n := seen[name]) > 0:
            value = value.name(f"{name}_{n}")
        seen[name] += 1
        out.append(value)
    return out


def _scan(table, scope):
    """Bind a scan of ``table``; a relation scanned twice gets a distinct view."""
    op = table.op()
    if op in scope.root.seen:
        return table.view()
    scope.root.seen.add(op)
    return table


def _bind_source(node, scope):
    """Convert a FROM or JOIN source and bind its alias; returns (alias, table)."""
    for arg, what in (
        ("pivots", "PIVOT"),
        ("joins", "A nested join"),
        ("sample", "TABLESAMPLE"),
    ):
        if node.args.get(arg):
            raise _unsupported(node, scope, what)
    if isinstance(node, sge.Table):
        name = node.name
        if (table := scope.cte(name)) is None:
            if (table := scope.root.catalog.get(name)) is None:
                raise com.IbisInputError(f"Table {name!r} is not in the catalog")
    elif isinstance(node, sge.Subquery):
        table = convert_relation(node.this, scope.child())
    else:
        raise _unsupported(node, scope)
    table = _scan(table, scope)
    alias = node.alias_or_name
    scope.bind(alias, table)
    return alias, table


_JOIN_SIDES = {"": "inner", "LEFT": "left", "RIGHT": "right", "FULL": "outer"}
_JOIN_KINDS = {
    "": None,
    "INNER": None,
    "OUTER": None,
    "CROSS": None,
    "SEMI": "semi",
    "ANTI": "anti",
}


def _hoistable(condition, visible, alias) -> bool:
    """Whether a WHERE condition is a join condition of the table ``alias``."""
    tables = _tables(condition)
    return (
        len(tables) >= 2
        and alias in tables
        and tables <= visible
        and condition.find(sge.Select) is None
    )


def _map_member(member, right, rel) -> None:
    """Address the columns of a joined table through the join chain's output."""
    columns = {}
    source = right.op()
    for name, value in rel.op().values.items():
        # a member is wrapped in a ``JoinReference`` unless it is a reference
        # already (a ``.view()``)
        if isinstance(value, ops.Field) and (
            value.rel == source
            or (isinstance(value.rel, ops.JoinReference) and value.rel.parent == source)
        ):
            columns[value.name] = name
    for name in right.columns:
        # an inner join key is not repeated in the output when it is equal to
        # the same-named column of the left side: it is that column
        columns.setdefault(name, name)
    member.columns = columns


def _build_from(select, scope):
    """Bind the FROM clause; returns the relation and the conditions to filter by."""
    where = _conjuncts(select.args.get("where"))
    if (from_ := _arg(select, "from_", "from")) is None:
        return None, where
    joins = select.args.get("joins") or []
    alias, rel = _bind_source(from_.this, scope)
    scope.rel = rel
    bound = {alias}
    deferred = []
    for i, join in enumerate(joins):
        if (
            join.method
            or join.args.get("using")
            or join.args.get("match_condition")
            or join.side not in _JOIN_SIDES
            or join.kind not in _JOIN_KINDS
        ):
            raise _unsupported(join, scope, "This kind of join")
        how = _JOIN_KINDS[join.kind] or _JOIN_SIDES[join.side]
        alias, right = _bind_source(join.this, scope)
        visible = bound | {alias}
        on = _conjuncts(join.args.get("on"))
        if (
            not on
            and join.kind in ("", "CROSS")
            and not join.side
            and all(later.side not in ("RIGHT", "FULL") for later in joins[i + 1 :])
        ):
            # ``FROM a, b WHERE a.x = b.y``: the conditions relating the new
            # table to the ones before it are its join conditions
            on = [c for c in where if _hoistable(c, visible, alias)]
            where = [c for c in where if not any(c is h for h in on)]
        local, late = [], []
        for condition in on:
            if condition.find(sge.Select) is not None or not (
                _tables(condition) <= visible
            ):
                # a subquery or an outer reference: a filter after the join
                late.append(condition)
            else:
                local.append(condition)
        if late and how != "inner":
            raise _unsupported(
                join, scope, "A subquery or outer reference in an outer join condition"
            )
        deferred.extend(late)
        _reject_aggregates(on, scope, "a join condition")
        predicates = [_boolean(condition, scope) for condition in local]
        rname = "{name}_" + alias
        if predicates:
            rel = rel.join(right, predicates, how=how, rname=rname)
        elif how == "inner":
            rel = rel.cross_join(right, rname=rname)
        else:
            rel = rel.join(right, how=how, rname=rname)
        _map_member(scope.members[alias], right, rel)
        scope.rebind(rel)
        bound = visible
    return rel, deferred + where


# relations the SQL compiler may merge into their parent SELECT, which loses
# the alias a correlated scalar subquery refers to
_MERGEABLE = (
    ops.Project,
    ops.Filter,
    ops.Sort,
    ops.Distinct,
    ops.DropColumns,
    ops.FillNull,
    ops.DropNull,
)


def _anchor(scope, nodes) -> None:
    """Give the relation a stable alias if ``nodes`` correlate a scalar subquery to it."""
    if scope.rel is None or not isinstance(scope.rel.op(), _MERGEABLE):
        return
    for root in nodes:
        for select in root.find_all(sge.Select):
            node = select.parent
            while node is not None:
                if isinstance(node, (sge.Exists, sge.In, sge.Any, sge.All)):
                    # these keep the parent SELECT from being merged
                    break
                if node is root:
                    if any(
                        column.table in scope.members
                        for column in select.find_all(sge.Column)
                    ):
                        scope.rebind(scope.rel.view())
                        return
                    break
                node = node.parent


def _metric_units(roots, scope):
    """Aggregate calls of this SELECT (with any FILTER) that are not window functions."""
    units = []
    for root in roots:
        for node in root.walk(
            prune=lambda n, root=root: (
                n is not root and isinstance(n, (sge.Select, sge.Subquery))
            )
        ):
            if not isinstance(node, sge.AggFunc) or _is_window_function(node):
                continue
            unit = node.parent if isinstance(node.parent, sge.Filter) else node
            for inner in unit.walk(
                prune=lambda n: isinstance(n, (sge.Select, sge.Subquery))
            ):
                if inner is not node and isinstance(inner, (sge.AggFunc, sge.Window)):
                    raise _invalid(unit, scope, "Aggregate functions cannot be nested")
            if unit not in units:
                units.append(unit)
    return units


def _aggregate(select, scope, units) -> None:
    """Aggregate the relation; later clauses find their expressions by lookup."""
    group = select.args.get("group")
    expressions = list(group.expressions) if group is not None else []
    aliases = {}
    for item in select.expressions:
        if isinstance(item, sge.Alias):
            aliases.setdefault(_unparen(item.this), item.alias)
    taken = set()

    def fresh(name):
        base, n = name, 0
        while name in taken:
            n += 1
            name = f"{base}_{n}"
        taken.add(name)
        return name

    memo, group_columns, groups = {}, {}, []
    for i, expression in enumerate(expressions):
        expression = _unparen(expression)
        if isinstance(expression, sge.Column):
            name = fresh(expression.name)
            group_columns[(expression.table, expression.name)] = name
        else:
            name = fresh(aliases.get(expression, f"_group_{i}"))
        groups.append(convert(expression, scope).name(name))
        memo[expression] = name
    metrics = []
    for i, unit in enumerate(units):
        alias = aliases.get(unit)
        name = fresh(alias if alias is not None and alias not in taken else f"_agg_{i}")
        metrics.append(convert(unit, scope).name(name))
        memo[unit] = name
    rel = scope.rel.aggregate(metrics, by=groups)
    scope.agg = _Aggregation(memo, group_columns)
    scope.rebind(rel)


def _has_bare_reduction(op) -> bool:
    if isinstance(op, ops.Reduction):
        return True
    if not isinstance(op, ops.Value) or isinstance(
        op, (ops.WindowFunction, ops.Subquery)
    ):
        return False
    return any(_has_bare_reduction(child) for child in op.__children__)


def _check_no_reductions(values, select, scope) -> None:
    # an aggregate the SELECT did not account for would be turned into a window
    # function or a scalar subquery by Ibis instead of raising
    for value in values:
        if _has_bare_reduction(value.op()):
            raise com.IbisError(
                "parse_sql did not account for an aggregate in "
                f"{select.sql(dialect=scope.root.dialect)}"
            )


def _finish(rel):
    return ir.Table(rel.op()) if isinstance(rel, ir.Join) else rel


def _is_plain(value) -> bool:
    """Whether ``value`` is free of window functions, aggregates and subqueries."""

    def plain(op):
        if isinstance(op, (ops.WindowFunction, ops.Reduction, ops.Subquery)):
            return False
        return not isinstance(op, ops.Value) or all(map(plain, op.__children__))

    return plain(value.op())


def _is_identity(values, rel) -> bool:
    """Whether ``values`` are exactly the columns of ``rel``, in order."""
    if len(values) != len(rel.columns):
        return False
    for value, name in zip(values, rel.columns):
        if value.get_name() != name:
            return False
        op = value.op()
        if isinstance(op, ops.Alias):
            op = op.arg
        if not (isinstance(op, ops.Field) and op.rel == rel.op() and op.name == name):
            return False
    return True


def _project(rel, values, elide=True):
    if elide and _is_identity(values, rel):
        return rel
    if isinstance(rel, ir.Join) and all(_is_plain(value) for value in values):
        # a plain projection of the join's members is the join's own output
        return rel.select(values)
    return _finish(rel).select(values)


def _order_by(order, scope, proj, values, distinct):
    scope.outputs = {value.get_name(): value for value in values}
    keys = [convert(key, scope) for key in order.expressions]
    try:
        return proj.order_by(keys)
    except com.IntegrityError:
        if distinct:
            raise _unsupported(
                order, scope, "ORDER BY a column that SELECT DISTINCT does not return"
            ) from None
    # a sort key the SELECT list does not carry (``SELECT a FROM t ORDER BY
    # c``): sort the projection extended with the keys, then drop them
    names = [value.get_name() for value in values]
    extra, resorted = [], []
    for i, key in enumerate(keys):
        op = key.op()
        name = f"_sort_{i}"
        extra.append(op.arg.to_expr().name(name))
        resorted.append((name, op.ascending, op.nulls_first))
    extended = _finish(scope.rel).select(values + extra)
    return extended.order_by(
        [
            ibis.asc(extended[name], nulls_first=nulls_first)
            if ascending
            else ibis.desc(extended[name], nulls_first=nulls_first)
            for name, ascending, nulls_first in resorted
        ]
    ).select(names)


def _limit(node, rel, scope):
    limit = node.args.get("limit")
    offset = node.args.get("offset")
    if limit is None and offset is None:
        return rel
    n = None
    if limit is not None:
        options = limit.args.get("limit_options")
        for option in ("percent", "with_ties"):
            if limit.args.get(option) or (options and options.args.get(option)):
                raise _unsupported(limit, scope, "LIMIT with PERCENT or WITH TIES")
        if isinstance(limit, sge.Fetch):
            # ``FETCH FIRST n ROWS ONLY`` is ``LIMIT n``; without ``n`` it is one row
            count = limit.args.get("count")
            n = 1 if count is None else _int(count, scope)
        else:
            n = _int(limit.expression, scope)
    return rel.limit(n, offset=0 if offset is None else _int(offset.expression, scope))


_UNSUPPORTED_SELECT_ARGS = {
    "qualify": "QUALIFY",
    "laterals": "LATERAL",
    "pivots": "PIVOT",
    "sample": "TABLESAMPLE",
    "locks": "FOR UPDATE",
    "connect": "CONNECT BY",
    "match": "MATCH_RECOGNIZE",
    "prewhere": "PREWHERE",
    "distribute": "DISTRIBUTE BY",
    "sort": "SORT BY",
    "cluster": "CLUSTER BY",
    "into": "SELECT INTO",
}
_UNSUPPORTED_GROUP_ARGS = {
    "rollup": "ROLLUP",
    "cube": "CUBE",
    "grouping_sets": "GROUPING SETS",
    "totals": "WITH TOTALS",
    "all": "GROUP BY ALL",
}


def _reject_unsupported(select, scope) -> None:
    for arg, what in _UNSUPPORTED_SELECT_ARGS.items():
        if value := select.args.get(arg):
            node = value if isinstance(value, sge.Expression) else select
            raise _unsupported(node, scope, what)
    if (distinct := select.args.get("distinct")) is not None and distinct.args.get(
        "on"
    ):
        raise _unsupported(distinct, scope, "DISTINCT ON")
    if (group := select.args.get("group")) is not None:
        for arg, what in _UNSUPPORTED_GROUP_ARGS.items():
            if group.args.get(arg):
                raise _unsupported(group, scope, what)


def _bind_ctes(node, scope) -> None:
    if (with_ := _arg(node, "with_", "with")) is None:
        return
    if with_.args.get("recursive"):
        raise _unsupported(with_, scope, "WITH RECURSIVE")
    for cte in with_.expressions:
        if cte.alias in scope.ctes:
            raise _invalid(cte, scope, f"The CTE {cte.alias!r} is defined twice")
        table = convert_relation(cte.this, scope.child())
        alias = cte.args.get("alias")
        columns = [column.name for column in alias.columns] if alias is not None else []
        if columns and columns != list(table.columns):
            table = table.rename(dict(zip(columns, table.columns)))
        scope.ctes[cte.alias] = table


def convert_relation(node, scope, **kwargs):
    """Convert a query (SELECT, set operation or parenthesized query) to a table."""
    if isinstance(node, sge.Select):
        return convert_select(node, scope, **kwargs)
    if isinstance(node, sge.SetOperation):
        return convert_set_operation(node, scope)
    if isinstance(node, sge.Subquery):
        return convert_relation(node.this, scope, **kwargs)
    raise _unsupported(node, scope)


def convert_select(select, scope, *, projection=True, elide=True):
    """Convert a SELECT against ``scope``.

    With ``projection=False`` only the rows are kept (``EXISTS``). With
    ``elide=False`` a projection is built even when it selects every column
    (a scalar subquery needs one).
    """
    _reject_unsupported(select, scope)
    _bind_ctes(select, scope)
    scope.windows = {
        window.this.name: window for window in select.args.get("windows") or ()
    }
    _reject_aggregates(_conjuncts(select.args.get("where")), scope, "WHERE")
    if (group := select.args.get("group")) is not None:
        _reject_aggregates(group.expressions, scope, "GROUP BY")
    rel, conditions = _build_from(select, scope)
    if rel is None:
        return _convert_from_less(select, scope)
    if conditions:
        _anchor(scope, conditions)
        scope.rebind(
            scope.rel.filter([_boolean(condition, scope) for condition in conditions])
        )

    items = [
        (item.this if isinstance(item, sge.Alias) else item, item.alias_or_name)
        for item in select.expressions
    ]
    having = select.args.get("having")
    order = select.args.get("order")
    roots = [expression for expression, _ in items]
    if having is not None:
        roots.append(having.this)
    if order is not None:
        roots.extend(key.this for key in order.expressions)
    units = _metric_units(roots, scope)
    if group is not None or units:
        _aggregate(select, scope, units)
    if having is not None:
        _reject_aggregates([having.this], scope, "HAVING", aggregates=False)
        _anchor(scope, [having.this])
        scope.rebind(scope.rel.filter(_boolean(having.this, scope)))
    if not projection:
        return scope.rel

    _anchor(scope, [expression for expression, _ in items])
    values = _unique_names(
        [convert(expression, scope).name(name) for expression, name in items]
    )
    _check_no_reductions(values, select, scope)
    distinct = select.args.get("distinct") is not None
    rel = _project(scope.rel, values, elide=elide)
    if distinct:
        rel = rel.distinct()
    if order is not None:
        rel = _order_by(order, scope, rel, values, distinct)
    return _limit(select, rel, scope)


def _convert_from_less(select, scope):
    # ``SELECT 1 AS x`` has no FROM clause
    values = {}
    for item in select.expressions:
        expression = item.this if isinstance(item, sge.Alias) else item
        op = convert(expression, scope).op()
        values[item.alias_or_name] = op.arg if isinstance(op, ops.Alias) else op
    rel = ops.DummyTable(values).to_expr()
    scope.rebind(rel)
    if where := _conjuncts(select.args.get("where")):
        rel = rel.filter([_boolean(condition, scope) for condition in where])
    if (order := select.args.get("order")) is not None:
        columns = [rel[name] for name in rel.columns]
        rel = _order_by(order, scope, rel, columns, distinct=False)
    return _limit(select, rel, scope)


_set_operations = {
    sge.Union: "union",
    sge.Intersect: "intersect",
    sge.Except: "difference",
}


def _align_set_operands(left, right):
    """SQL set operations match columns by position; Ibis matches by name."""
    from ibis.expr.datatypes.cast import highest_precedence

    left_values, right_values = [], []
    for name, rname in zip(left.columns, right.columns):
        lcol, rcol = left[name], right[rname]
        target = highest_precedence([lcol.type(), rcol.type()])
        if lcol.type() != target:
            lcol = lcol.cast(target)
        if rcol.type() != target:
            rcol = rcol.cast(target)
        left_values.append(lcol.name(name))
        right_values.append(rcol.name(name))
    return left.select(left_values), right.select(right_values)


def convert_set_operation(setop, scope):
    _bind_ctes(setop, scope)
    if setop.args.get("by_name"):
        raise _unsupported(setop, scope, "A set operation BY NAME")
    if (method := _set_operations.get(type(setop))) is None:
        raise _unsupported(setop, scope)
    left = convert_relation(setop.left, scope.child())
    right = convert_relation(setop.right, scope.child())
    if len(left.columns) != len(right.columns):
        raise _invalid(
            setop,
            scope,
            f"The sides of the set operation return {len(left.columns)} "
            f"and {len(right.columns)} columns",
        )
    if left.schema() != right.schema():
        try:
            left, right = _align_set_operands(left, right)
        except com.IbisTypeError as e:
            raise _invalid(setop, scope, f"Cannot align the sides: {e}") from e
    table = getattr(left, method)(right, distinct=bool(setop.args.get("distinct")))

    if (order := setop.args.get("order")) is not None:
        keys = scope.child()
        keys.rel = table
        keys.outputs = {name: table[name] for name in table.columns}
        table = table.order_by([convert(key, keys) for key in order.expressions])
    return _limit(setop, table, scope)


@singledispatch
def _convert(node, scope):
    raise _unsupported(node, scope)


def convert(node, scope):
    """Convert a value expression against ``scope``."""
    if scope.agg is not None and (name := scope.agg.memo.get(node)) is not None:
        # an aggregate or GROUP BY expression computed by the aggregate
        return scope.rel[name]
    try:
        return _convert(node, scope)
    except (TypeError, ValidationError, AttributeError) as e:
        # Ibis rejected the operands, or a rule called a method the operand's
        # type lacks (``LENGTH(NULL)``); say which SQL they came from
        raise _invalid(node, scope, f"Cannot translate ({e})") from e


@_convert.register(sge.Subquery)
def convert_subquery(subquery, scope):
    table = convert_relation(subquery.this, scope.child(), elide=False)
    if len(table.columns) != 1:
        raise com.IbisInputError(
            f"A scalar subquery must return one column: {subquery.sql(dialect=scope.root.dialect)}"
        )
    return table.as_scalar()


@_convert.register(sge.Exists)
def convert_exists(exists, scope):
    rel = convert_relation(exists.this, scope.child(), projection=False)
    if isinstance(rel.op(), (ops.PhysicalTable, ops.Reference)):
        # the compiler puts the subquery's SELECT inside EXISTS; a bare table
        # has none
        rel = rel.select(rel.columns)
    return ops.ExistsSubquery(rel.op()).to_expr()


def _single_column(node, scope):
    table = convert_relation(node, scope.child())
    if len(table.columns) != 1:
        raise com.IbisInputError(
            f"The subquery must return one column: {node.sql(dialect=scope.root.dialect)}"
        )
    (name,) = table.columns
    return table[name]


@_convert.register(sge.Filter)
def convert_filtered_aggregate(node, scope):
    value = convert(node.this, scope)
    if not isinstance(op := value.op(), ops.Reduction):
        raise _unsupported(node, scope, "FILTER on a non-aggregate function")
    where = convert(node.expression.this, scope)
    return op.copy(where=where.op()).to_expr()


def _number(text):
    """A non-integer numeric literal.

    Written with a decimal point it is a decimal typed by its digits, as SQL
    engines type it (``1.10`` is ``decimal(3, 2)``); written with an exponent
    it is a float.
    """
    if "e" in text.lower():
        return ibis.literal(float(text))
    whole, _, fraction = text.lstrip("-").partition(".")
    precision = len(whole or "0") + len(fraction)
    if precision > 38:
        return ibis.literal(float(text))
    return ibis.literal(
        decimal.Decimal(text), type=dt.Decimal(precision, len(fraction))
    )


@_convert.register(sge.Literal)
def convert_literal(literal, scope):
    value = literal.this
    if literal.is_int:
        return ibis.literal(int(value))
    if literal.is_number:
        return _number(value)
    return ibis.literal(value)


@_convert.register(sge.Boolean)
def convert_boolean(boolean, scope):
    return ibis.literal(boolean.this)


@_convert.register(sge.Null)
def convert_null(null, scope):
    return ibis.null()


@_convert.register(sge.Alias)
def convert_alias(alias, scope):
    this = convert(alias.this, scope=scope)
    return this.name(alias.alias)


@_convert.register(sge.Column)
def convert_column(column, scope):
    return scope.column(column.table, column.name)


@_convert.register(sge.Ordered)
def convert_ordered(ordered, scope):
    this = convert(_unparen(ordered.this), scope=scope)
    desc = ordered.args.get("desc", False)  # not exposed as an attribute
    nulls_first = ordered.args.get("nulls_first", False)
    return (
        ibis.desc(this, nulls_first=nulls_first)
        if desc
        else ibis.asc(this, nulls_first=nulls_first)
    )


_unary_operations = {
    sge.Paren: lambda x: x,
    sge.Not: operator.invert,
    sge.Neg: operator.neg,
}


@_convert.register(sge.Unary)
def convert_unary(unary, scope):
    if (op := _unary_operations.get(type(unary))) is None:
        raise _unsupported(unary, scope)
    this = convert(unary.this, scope=scope)
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
    # SQL's integer division truncates toward zero where Ibis's ``//`` floors;
    # taking the remainder out first makes the quotient exact for both
    sge.IntDiv: lambda left, right: (left - left % right) // right,
    sge.Mod: operator.mod,
    sge.Pow: operator.pow,
    sge.And: operator.and_,
    sge.Or: operator.or_,
    # IS [NOT] DISTINCT FROM
    sge.NullSafeEQ: lambda left, right: left.identical_to(right),
    sge.NullSafeNEQ: lambda left, right: ~left.identical_to(right),
}


_COMPARISONS = (sge.LT, sge.LTE, sge.GT, sge.GTE, sge.EQ, sge.NEQ)
_NULL_SAFE = (sge.NullSafeEQ, sge.NullSafeNEQ)
_ARITHMETIC = (sge.Add, sge.Sub, sge.Mul, sge.Div, sge.IntDiv, sge.Mod, sge.Pow)


def _align_numeric(left, right):
    """Give the operands of an arithmetic operator a type Ibis can rank.

    A float and a decimal compute in floating point in SQL engines, where
    Ibis would type the result decimal and narrow it to that scale on fetch.
    Two decimals rank by precision and scale together, so one narrower in
    precision but wider in scale (``decimal(38, 2) * 0.0001``) has no common
    type until the narrower one is widened to both.
    """
    ltype, rtype = left.type(), right.type()
    if ltype.is_floating() and rtype.is_decimal():
        return left, right.cast(ltype)
    if rtype.is_floating() and ltype.is_decimal():
        return left.cast(rtype), right
    if (
        ltype.is_decimal()
        and rtype.is_decimal()
        and None not in (ltype.precision, ltype.scale, rtype.precision, rtype.scale)
        and not (ltype.castable(rtype) or rtype.castable(ltype))
    ):
        wide = dt.Decimal(
            min(max(ltype.precision, rtype.precision), 38),
            max(ltype.scale, rtype.scale),
        )
        if ltype.precision < rtype.precision:
            return left.cast(wide), right
        return left, right.cast(wide)
    return left, right


def _convert_quantified(binary, scope):
    quantifier = binary.expression
    if isinstance(binary, sge.EQ) and isinstance(quantifier, sge.Any):
        return convert(binary.this, scope).isin(_single_column(quantifier.this, scope))
    if isinstance(binary, sge.NEQ) and isinstance(quantifier, sge.All):
        return ~convert(binary.this, scope).isin(_single_column(quantifier.this, scope))
    raise _unsupported(binary, scope, "ANY or ALL with this operator")


@_convert.register(sge.Binary)
def convert_binary(binary, scope):
    if (op := _binary_operations.get(type(binary))) is None:
        raise _unsupported(binary, scope)
    left, right = binary.this, binary.expression
    if isinstance(right, (sge.Any, sge.All)):
        return _convert_quantified(binary, scope)
    if (isinstance(left, sge.Null) or isinstance(right, sge.Null)) and not isinstance(
        binary, _NULL_SAFE
    ):
        # SQL's NULL compares to and computes with everything as NULL; Ibis
        # would read ``x = NULL`` as ``x IS NULL``
        if isinstance(binary, _COMPARISONS):
            return ibis.null(dt.boolean)
        if isinstance(binary, (sge.And, sge.Or)):
            null = ibis.null(dt.boolean)
            return op(
                null if isinstance(left, sge.Null) else convert(left, scope),
                null if isinstance(right, sge.Null) else convert(right, scope),
            )
        other = convert(right if isinstance(left, sge.Null) else left, scope)
        return ibis.null(other.type())
    left, right = convert(left, scope), convert(right, scope)
    if isinstance(binary, _ARITHMETIC):
        left, right = _align_numeric(left, right)
    return op(left, right)


@_convert.register(sge.Is)
def convert_is(is_, scope):
    this = convert(is_.this, scope=scope)
    if isinstance(is_.expression, sge.Null):
        return this.isnull()
    if isinstance(is_.expression, sge.Boolean):
        # ``IS TRUE`` and ``IS FALSE`` are false for NULL where ``=`` would be
        # NULL; ``IS NOT TRUE`` is the ``Not`` of ``IS TRUE`` and follows
        return (this if is_.expression.this else ~this).fill_null(False)
    raise _unsupported(is_, scope)


def _convert_like(like, scope, method):
    this = convert(like.this, scope=scope)
    result = getattr(this, method)(convert(like.expression, scope=scope))
    # sqlglot parses ``x NOT LIKE y`` as ``Like(negate=True)`` from 30.8 and as
    # ``Not(Like)`` before; the argument is absent on the older releases
    return ~result if like.args.get("negate") else result


@_convert.register(sge.Like)
def convert_like(like, scope):
    return _convert_like(like, scope, "like")


@_convert.register(sge.ILike)
def convert_ilike(like, scope):
    return _convert_like(like, scope, "ilike")


@_convert.register(sge.Between)
def convert_between(between, scope):
    if between.args.get("symmetric"):
        raise _unsupported(between, scope, "BETWEEN SYMMETRIC")
    this = convert(between.this, scope=scope)
    low = convert(between.args["low"], scope=scope)
    high = convert(between.args["high"], scope=scope)
    return this.between(low, high)


_reduction_methods = {
    sge.Max: "max",
    sge.Min: "min",
    sge.Quantile: "quantile",
    sge.Sum: "sum",
    sge.Avg: "mean",
    sge.ArrayAgg: "collect",
    sge.LogicalOr: "any",
    sge.LogicalAnd: "all",
}


@_convert.register(sge.AggFunc)
def convert_reduction(reduction, scope):
    if (method := _reduction_methods.get(type(reduction))) is None:
        raise _unsupported(reduction, scope)
    this = convert(reduction.this, scope=scope)
    return getattr(this, method)()


@_convert.register(sge.Stddev)
@_convert.register(sge.StddevSamp)
def convert_stddev_samp(reduction, scope):
    return convert(reduction.this, scope=scope).std(how="sample")


@_convert.register(sge.StddevPop)
def convert_stddev_pop(reduction, scope):
    return convert(reduction.this, scope=scope).std(how="pop")


@_convert.register(sge.Variance)
def convert_variance_samp(reduction, scope):
    return convert(reduction.this, scope=scope).var(how="sample")


@_convert.register(sge.VariancePop)
def convert_variance_pop(reduction, scope):
    return convert(reduction.this, scope=scope).var(how="pop")


@_convert.register(sge.ApproxDistinct)
def convert_approx_distinct(reduction, scope):
    return convert(reduction.this, scope=scope).approx_nunique()


@_convert.register(sge.GroupConcat)
def convert_group_concat(func, scope):
    this = func.this
    order_by = None
    if isinstance(this, sge.Order):
        # ``STRING_AGG(x, ',' ORDER BY y)``
        order_by = [convert(key, scope=scope) for key in this.expressions]
        this = this.this
    if (separator := func.args.get("separator")) is None:
        sep = ","
    elif isinstance(separator, sge.Literal) and separator.is_string:
        sep = separator.name
    else:
        raise _unsupported(func, scope, "STRING_AGG with a computed separator")
    return convert(this, scope=scope).group_concat(sep, order_by=order_by)


# the Ibis function for each ranking function, by the name of its sqlglot
# class; ``Rank``, ``DenseRank``, ``PercentRank``, ``CumeDist`` and ``Ntile``
# exist from sqlglot 27.11, so they are registered where ``sge`` has them
_ranking_functions = {
    "RowNumber": "row_number",
    "Rank": "rank",
    "DenseRank": "dense_rank",
    "PercentRank": "percent_rank",
    "CumeDist": "cume_dist",
}


def _register(name):
    """Register a rule for the sqlglot class ``name`` if this sqlglot has it."""
    cls = getattr(sge, name, None)
    return _convert.register(cls) if cls is not None else (lambda f: f)


@_register("RowNumber")
@_register("Rank")
@_register("DenseRank")
@_register("PercentRank")
@_register("CumeDist")
def convert_ranking(func, scope):
    name = type(func).__name__
    result = getattr(ibis, _ranking_functions[name])()
    if name in ("RowNumber", "Rank", "DenseRank"):
        # SQL's ranking functions are 1-based, Ibis's are 0-based
        return result + 1
    return result


@_register("Ntile")
def convert_ntile(ntile, scope):
    # SQL's NTILE is 1-based, Ibis's is 0-based
    return ibis.ntile(convert(ntile.this, scope=scope)) + 1


@_convert.register(sge.Lag)
@_convert.register(sge.Lead)
def convert_lag_lead(func, scope):
    this = convert(func.this, scope=scope)
    offset = func.args.get("offset")
    default = func.args.get("default")
    method = "lag" if isinstance(func, sge.Lag) else "lead"
    return getattr(this, method)(
        None if offset is None else convert(offset, scope=scope),
        None if default is None else convert(default, scope=scope),
    )


@_convert.register(sge.FirstValue)
def convert_first_value(func, scope):
    return convert(func.this, scope=scope).first()


@_convert.register(sge.LastValue)
def convert_last_value(func, scope):
    return convert(func.this, scope=scope).last()


@_convert.register(sge.NthValue)
def convert_nth_value(func, scope):
    this = convert(func.this, scope=scope)
    # SQL's NTH_VALUE is 1-based, ``nth`` is 0-based
    return this.nth(convert(func.args["offset"], scope=scope) - 1)


def _window_bound(value, side):
    """Translate a window frame bound into Ibis's signed offset convention."""
    if value is None:
        return None
    text = value if isinstance(value, str) else value.name
    if text.upper() == "UNBOUNDED":
        return None
    if text.upper() == "CURRENT ROW":
        return 0
    offset = int(text)
    return -offset if (side or "").upper() == "PRECEDING" else offset


def _window_parts(window, scope):
    """The PARTITION BY, ORDER BY and frame of ``window``.

    ``OVER w`` takes all three from the WINDOW clause; ``OVER (w ...)`` may add
    an ORDER BY and a frame to a named window that has none, not a partition.
    """
    parts = {name: window.args.get(name) for name in ("partition_by", "order", "spec")}
    if (alias := window.args.get("alias")) is None:
        return parts
    if (named := scope.windows.get(alias.name)) is None:
        raise _unsupported(window, scope, f"The window {alias.name!r}")
    if parts["partition_by"]:
        raise _invalid(
            window, scope, f"Cannot override the PARTITION BY of {alias.name!r}"
        )
    if any(parts.values()) and named.args.get("spec") is not None:
        raise _invalid(
            window, scope, f"Cannot extend the window {alias.name!r}: it has a frame"
        )
    if parts["order"] is not None and named.args.get("order") is not None:
        raise _invalid(window, scope, f"Cannot override the ORDER BY of {alias.name!r}")
    return {name: part or named.args.get(name) for name, part in parts.items()}


@_convert.register(sge.Window)
def convert_window(window, scope):
    parts = _window_parts(window, scope)
    func = convert(window.this, scope=scope)
    group_by = [convert(part, scope=scope) for part in parts["partition_by"] or ()]
    order_by = []
    if (order := parts["order"]) is not None:
        order_by = [convert(key, scope=scope) for key in order.expressions]

    kwargs = {}
    if (spec := parts["spec"]) is not None:
        if spec.args.get("exclude"):
            raise _unsupported(window, scope, "A window frame with EXCLUDE")
        start = _window_bound(spec.args.get("start"), spec.args.get("start_side"))
        if (end := spec.args.get("end")) is None:
            # ``ROWS 1 PRECEDING`` names the start only; the end is the current row
            end = 0
        else:
            end = _window_bound(end, spec.args.get("end_side"))
        kind = (spec.args.get("kind") or "ROWS").upper()
        kwargs["rows" if kind == "ROWS" else "range"] = (start, end)
    else:
        # a FILTER wraps the aggregate the window applies to
        this = window.this.this if isinstance(window.this, sge.Filter) else window.this
        if order_by and isinstance(this, sge.AggFunc):
            # SQL's default frame for an ordered aggregate window ends at the
            # current row (and its peers); Ibis's default is the whole partition
            kwargs["range"] = (None, 0)

    result = func.over(ibis.window(group_by=group_by, order_by=order_by, **kwargs))
    if isinstance(result, Deferred):
        # neither the function nor the window names a table (``ROW_NUMBER()
        # OVER ()``): the window is over the relation of this SELECT
        if scope.rel is None:
            raise _unsupported(window, scope, "A window function without a FROM")
        result = result.resolve(scope.rel)
    return result


_truncation_units = {
    "year": "Y",
    "quarter": "Q",
    "month": "M",
    "week": "W",
    "day": "D",
    "hour": "h",
    "minute": "m",
    "second": "s",
    "millisecond": "ms",
    "microsecond": "us",
}


@_convert.register(sge.TimestampTrunc)
@_convert.register(sge.DateTrunc)
def convert_truncate(trunc, scope):
    this = convert(trunc.this, scope=scope)
    return this.truncate(_unit(trunc.args.get("unit"), _truncation_units, trunc, scope))


@_convert.register(sge.Round)
def convert_round(func, scope):
    this = convert(func.this, scope=scope)
    if (decimals := func.args.get("decimals")) is not None:
        return this.round(convert(decimals, scope=scope))
    return this.round()


@_convert.register(sge.Abs)
def convert_abs(func, scope):
    return convert(func.this, scope=scope).abs()


@_convert.register(sge.Floor)
def convert_floor(func, scope):
    return convert(func.this, scope=scope).floor()


@_convert.register(sge.Ceil)
def convert_ceil(func, scope):
    return convert(func.this, scope=scope).ceil()


@_convert.register(sge.Sqrt)
def convert_sqrt(func, scope):
    return convert(func.this, scope=scope).sqrt()


@_convert.register(sge.Ln)
def convert_ln(func, scope):
    return convert(func.this, scope=scope).ln()


@_convert.register(sge.Exp)
def convert_exp(func, scope):
    return convert(func.this, scope=scope).exp()


@_convert.register(sge.Log)
def convert_log(func, scope):
    if (expression := func.expression) is None:
        # sqlglot parses ``LOG(x)`` as ``LN(x)`` for the dialects where it is
        # the natural logarithm, so a bare ``LOG`` is base 10
        return convert(func.this, scope=scope).log10()
    return convert(expression, scope=scope).log(convert(func.this, scope=scope))


@_convert.register(sge.Greatest)
def convert_greatest(func, scope):
    return ibis.greatest(
        convert(func.this, scope=scope),
        *(convert(e, scope=scope) for e in func.expressions),
    )


@_convert.register(sge.Least)
def convert_least(func, scope):
    return ibis.least(
        convert(func.this, scope=scope),
        *(convert(e, scope=scope) for e in func.expressions),
    )


@_convert.register(sge.Concat)
def convert_concat(func, scope):
    args = [convert(e, scope=scope) for e in func.expressions]
    if func.args.get("coalesce"):
        # in this dialect CONCAT() skips NULL arguments where ``||``
        # propagates them
        args = [ibis.coalesce(arg, "") for arg in args]
    first, *rest = args
    for arg in rest:
        first = first + arg
    return first


@_convert.register(sge.ConcatWs)
def convert_concat_ws(func, scope):
    sep, first, *rest = [convert(e, scope=scope) for e in func.expressions]
    # NULL arguments are skipped, so the separator goes between the arguments
    # that are present, and the result is '' when none is
    result = first
    for arg in rest:
        result = arg.isnull().ifelse(
            result, result.isnull().ifelse(arg, result + sep + arg)
        )
    return result.fill_null("")


@_convert.register(sge.DPipe)
def convert_dpipe(func, scope):
    return convert(func.this, scope=scope) + convert(func.expression, scope=scope)


@_convert.register(sge.Trim)
def convert_trim(func, scope):
    if func.expression is not None:
        raise _unsupported(func, scope, "TRIM of specific characters")
    this = convert(func.this, scope=scope)
    position = (func.args.get("position") or "BOTH").upper()
    return {"LEADING": this.lstrip, "TRAILING": this.rstrip}.get(position, this.strip)()


@_convert.register(sge.Initcap)
def convert_initcap(func, scope):
    return convert(func.this, scope=scope).capitalize()


@_convert.register(sge.If)
def convert_if(func, scope):
    cond = convert(func.this, scope=scope)
    true = convert(func.args["true"], scope=scope)
    false = func.args.get("false")
    return cond.ifelse(true, None if false is None else convert(false, scope=scope))


@_convert.register(sge.Count)
def convert_count(count, scope):
    this = count.this
    if this is None or isinstance(this, (sge.Star, sge.Literal, sge.Boolean)):
        # ``COUNT(1)`` counts rows like ``COUNT(*)``
        return scope.rel.count() if scope.rel is not None else ibis._.count()
    if isinstance(this, sge.Distinct):
        if len(this.expressions) != 1:
            raise _unsupported(count, scope, "COUNT(DISTINCT) of several expressions")
        (this,) = this.expressions
        return convert(this, scope=scope).nunique()
    return convert(this, scope=scope).count()


@_convert.register(sge.In)
def convert_in(in_, scope):
    this = convert(in_.this, scope=scope)
    if (query := in_.args.get("query")) is not None:
        return this.isin(_single_column(query, scope))
    candidates = [convert(expression, scope) for expression in in_.expressions]
    return this.isin(candidates)


@_convert.register(sge.Cast)
def convert_cast(cast, scope):
    this = convert(cast.this, scope)
    to = convert(cast.to, scope)

    return this.cast(to)


@_convert.register(sge.TryCast)
def convert_try_cast(cast, scope):
    return convert(cast.this, scope).try_cast(convert(cast.to, scope))


@_convert.register(sge.DataType)
def convert_datatype(datatype, scope):
    return SqlglotType().to_ibis(datatype)


@_convert.register(sge.Case)
def convert_case(case, scope):
    branches = [
        (
            convert(if_.this, scope=scope),
            convert(if_.args["true"], scope=scope),
        )
        for if_ in case.args["ifs"]
    ]
    default = case.args.get("default")
    else_ = None if default is None else convert(default, scope=scope)
    if (base := case.this) is not None:
        return convert(base, scope=scope).cases(*branches, else_=else_)
    return ibis.cases(*branches, else_=else_)


@_convert.register(sge.StrPosition)
def convert_str_position(func, scope):
    if func.args.get("position") is not None or func.args.get("occurrence"):
        raise _unsupported(func, scope, "POSITION with a start or occurrence")
    this = convert(func.this, scope=scope)
    # SQL positions are 1-based with 0 for "not found"; ``find`` is 0-based
    return this.find(convert(func.args["substr"], scope=scope)) + 1


@_convert.register(sge.SplitPart)
def convert_split_part(func, scope):
    index = func.args["part_index"]
    if (n := _literal_int(index)) is not None and n < 1:
        raise _unsupported(func, scope, "SPLIT_PART with a part before the first")
    this = convert(func.this, scope=scope)
    parts = this.split(convert(func.args["delimiter"], scope=scope))
    # SQL's parts are 1-based, and a part past the last one is the empty string
    part = parts[n - 1 if n is not None else convert(index, scope=scope) - 1]
    return this.isnull().ifelse(ibis.null(dt.string), part.fill_null(""))


@_convert.register(sge.RegexpLike)
def convert_regexp_like(func, scope):
    if func.args.get("flag") is not None:
        raise _unsupported(func, scope, "A regular expression with flags")
    this = convert(func.this, scope=scope)
    return this.re_search(convert(func.expression, scope=scope))


def _literal_int(node) -> int | None:
    """The value of an integer literal, possibly negated; else ``None``."""
    if isinstance(node, sge.Neg) and (value := _literal_int(node.this)) is not None:
        return -value
    if isinstance(node, sge.Literal) and node.is_int:
        return int(node.this)
    return None


@_convert.register(sge.Substring)
def convert_substring(substring, scope):
    this = convert(substring.this, scope=scope)
    if (start := _literal_int(substring.args["start"])) is not None and start < 1:
        # engines disagree on what a start before the first character means
        raise _unsupported(substring, scope, "SUBSTRING with a start before 1")
    # SQL substrings are 1-based
    start = convert(substring.args["start"], scope=scope) - 1
    if (length := substring.args.get("length")) is not None:
        return this.substr(start, convert(length, scope=scope))
    return this.substr(start)


_extract_methods = {
    "year": "year",
    "quarter": "quarter",
    "month": "month",
    "week": "week_of_year",
    "day": "day",
    "doy": "day_of_year",
    "dayofyear": "day_of_year",
    "hour": "hour",
    "minute": "minute",
    "second": "second",
    "millisecond": "millisecond",
    "microsecond": "microsecond",
    "epoch": "epoch_seconds",
}


def _extract(unit, this, node, scope):
    unit = unit.lower().removesuffix("s")
    if unit in ("dow", "dayofweek"):
        # SQL counts the days of the week from Sunday as 0, Ibis from Monday
        return (this.day_of_week.index() + 1) % 7
    if unit == "isodow":
        return this.day_of_week.index() + 1
    if (method := _extract_methods.get(unit)) is None:
        raise _unsupported(node, scope, f"Extracting {unit!r}")
    return getattr(this, method)()


@_convert.register(sge.Extract)
def convert_extract(extract, scope):
    return _extract(
        extract.this.name, convert(extract.expression, scope=scope), extract, scope
    )


@_convert.register(sge.DateDiff)
def convert_date_diff(func, scope):
    unit = func.args.get("unit")
    unit = "day" if unit is None else unit.name.lower().removesuffix("s")
    this = convert(func.this, scope=scope)
    return this.delta(convert(func.expression, scope=scope), unit=unit)


@_convert.register(sge.CurrentDate)
def convert_current_date(func, scope):
    return ibis.today()


@_convert.register(sge.CurrentTimestamp)
def convert_current_timestamp(func, scope):
    return ibis.now()


@_convert.register(sge.Anonymous)
def convert_anonymous(func, scope):
    # functions sqlglot leaves as calls by name
    name = func.name.upper()
    if name in ("DATE_PART", "DATEPART") and len(func.expressions) == 2:
        unit, this = func.expressions
        if isinstance(unit, sge.Literal) and unit.is_string:
            return _extract(unit.name, convert(this, scope=scope), func, scope)
    if name == "NOW" and not func.expressions:
        return ibis.now()
    if name == "TODAY" and not func.expressions:
        return ibis.today()
    raise _unsupported(func, scope, f"The function {func.name}")


_interval_units = {
    "year": "Y",
    "quarter": "Q",
    "month": "M",
    "week": "W",
    "day": "D",
    "hour": "h",
    "minute": "m",
    "second": "s",
}


def _unit(node, units, parent, scope):
    """The Ibis unit for a date or interval unit of the SQL."""
    if node is None or (unit := units.get(node.name.lower().removesuffix("s"))) is None:
        raise _unsupported(parent, scope, f"The unit {getattr(node, 'name', None)!r}")
    return unit


@_convert.register(sge.DateAdd)
@_convert.register(sge.DateSub)
def convert_date_add(func, scope):
    this = convert(func.this, scope=scope)
    unit = func.args.get("unit")
    if isinstance(func.expression, sge.Literal):
        unit = "D" if unit is None else _unit(unit, _interval_units, func, scope)
        interval = ibis.interval(int(func.expression.name), unit)
    else:
        # the argument may be an interval already (``DATE_ADD(d, INTERVAL 1
        # DAY)``) or a number of ``unit``
        interval = convert(func.expression, scope=scope)
        if not interval.type().is_interval():
            unit = "D" if unit is None else _unit(unit, _interval_units, func, scope)
            interval = interval.as_interval(unit)
    return this - interval if isinstance(func, sge.DateSub) else this + interval


@_convert.register(sge.Interval)
def convert_interval(interval, scope):
    unit = _unit(interval.args.get("unit"), _interval_units, interval, scope)
    if isinstance(interval.this, sge.Literal):
        return ibis.interval(int(interval.this.name), unit)
    return convert(interval.this, scope=scope).as_interval(unit)


@_convert.register(sge.Nullif)
def convert_nullif(nullif, scope):
    this = convert(nullif.this, scope=scope)
    return this.nullif(convert(nullif.expression, scope=scope))


@_convert.register(sge.Coalesce)
def convert_coalesce(coalesce, scope):
    args = [convert(coalesce.this, scope=scope)] + [
        convert(e, scope=scope) for e in coalesce.expressions
    ]
    return ibis.coalesce(*args)


@_convert.register(sge.Upper)
def convert_upper(upper, scope):
    return convert(upper.this, scope=scope).upper()


@_convert.register(sge.Lower)
def convert_lower(lower, scope):
    return convert(lower.this, scope=scope).lower()


@_convert.register(sge.Length)
def convert_length(length, scope):
    return convert(length.this, scope=scope).length()


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

    Notes
    -----
    A numeric literal written with a decimal point is typed ``decimal`` by its
    digits, as engines type it; written with an exponent it is ``float64``.
    A ``SUBSTRING`` whose start is a literal before 1 is rejected, because
    engines disagree on what it means; a computed start before 1 follows
    Ibis, which counts a negative start from the end of the string.

    """
    catalog = Catalog(
        {name: ibis.table(schema, name=name) for name, schema in catalog.items()}
    )

    try:
        statements = [s for s in sg.parse(sqlstring, dialect) if s is not None]
    except sg.errors.ParseError as e:
        raise com.IbisInputError(f"Cannot parse the SQL: {e}") from e
    if len(statements) != 1:
        raise com.IbisInputError(
            f"parse_sql takes one statement, {len(statements)} were given"
        )
    (expr,) = statements
    scope = Scope(_Root(catalog, dialect))
    known = set(catalog) | {cte.alias for cte in expr.find_all(sge.CTE)}
    for table in expr.find_all(sge.Table):
        if isinstance(table.this, sge.Identifier) and table.name not in known:
            raise com.IbisInputError(f"Table {table.name!r} is not in the catalog")
    for join in expr.find_all(sge.Join):
        if join.method:
            # qualify cannot resolve the columns of these joins; say why first
            raise _unsupported(join, scope, f"A {join.method} join")
    try:
        tree = sgo.optimize(
            expr,
            catalog.to_sqlglot(),
            dialect=dialect,
            rules=RULES,
            isolate_tables=False,
        )
    except sg.errors.OptimizeError as e:
        raise com.IbisInputError(str(e)) from e

    return convert_relation(tree, scope)


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
def to_sql(
    expr: ir.Expr, dialect: str | None = None, pretty: bool = True, **kwargs
) -> SQLString:
    """Return the formatted SQL string for an expression.

    Parameters
    ----------
    expr
        Ibis expression.
    dialect
        SQL dialect to use for compilation.
    pretty
        Whether to use pretty formatting.
    kwargs
        Scalar parameters

    Returns
    -------
    str
        Formatted SQL string

    Examples
    --------
    >>> import ibis
    >>> t = ibis.table({"a": "int", "b": "int"}, name="t")
    >>> expr = t.mutate(c=t.a + t.b)
    >>> ibis.to_sql(expr)  # doctest: +SKIP
    SELECT
      "t0"."a",
      "t0"."b",
      "t0"."a" + "t0"."b" AS "c"
    FROM "t" AS "t0"

    You can also specify the SQL dialect to use for compilation:
    >>> ibis.to_sql(expr, dialect="mysql")  # doctest: +SKIP
    SELECT
      `t0`.`a`,
      `t0`.`b`,
      `t0`.`a` + `t0`.`b` AS `c`
    FROM `t` AS `t0`

    See Also
    --------
    [`Table.compile()`](./expression-tables.qmd#ibis.expr.types.relations.Table.compile)

    """
    import ibis.backends.sql.compilers as sc

    # try to infer from a non-str expression or if not possible fallback to
    # the default pretty dialect for expressions
    if dialect is None:
        try:
            compiler_provider = expr._find_backend(use_default=True)
        except com.IbisError:
            # default to duckdb for SQL compilation because it supports the
            # widest array of ibis features for SQL backends
            compiler_provider = sc.duckdb
    else:
        try:
            compiler_provider = getattr(sc, dialect)
        except AttributeError as e:
            raise ValueError(f"Unknown dialect {dialect}") from e

    if (compiler := getattr(compiler_provider, "compiler", None)) is None:
        raise NotImplementedError(f"{compiler_provider} is not a SQL backend")

    out = compiler.to_sqlglot(expr.unbind(), **kwargs)
    queries = out if isinstance(out, list) else [out]
    dialect = compiler.dialect
    sql = ";\n".join(query.sql(dialect=dialect, pretty=pretty) for query in queries)
    return SQLString(sql)
