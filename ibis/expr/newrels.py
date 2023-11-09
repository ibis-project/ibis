from __future__ import annotations

from abc import abstractmethod
from typing import Annotated, Any

import ibis.expr.datashape as ds
import ibis.expr.datatypes as dt
import ibis.expr.types as ir
from ibis.common.annotations import attribute
from ibis.common.collections import FrozenDict  # noqa: TCH001
from ibis.common.deferred import Deferred, Item, deferred, var
from ibis.common.exceptions import IntegrityError
from ibis.common.graph import traverse
from ibis.common.patterns import Check, InstanceOf, _, pattern, replace
from ibis.common.typing import Coercible, VarTuple
from ibis.expr.operations import Alias, Column, Node, Scalar, SortKey, Value
from ibis.expr.schema import Schema
from ibis.expr.types import Expr, literal
from ibis.expr.types import Value as ValueExpr
from ibis.selectors import Selector
from ibis.util import Namespace

# need a parallel Expression and Operation class hierarchy to decompose ops.Selection
# into proper relational algebra operations


################################ OPERATIONS ################################


class Relation(Node, Coercible):
    @classmethod
    def __coerce__(cls, value):
        if isinstance(value, Relation):
            return value
        elif isinstance(value, TableExpr):
            return value.op()
        else:
            raise TypeError(f"Cannot coerce {value!r} to a Relation")

    @attribute
    def fields(self):
        raise NotImplementedError()

    @attribute
    def schema(self):
        return Schema({k: v.dtype for k, v in self.fields.items()})

    def to_expr(self):
        return TableExpr(self)


class Field(Value):
    rel: Relation
    name: str

    shape = ds.columnar

    @attribute
    def dtype(self):
        return self.rel.schema[self.name]


def _check_integrity(parent, values):
    possible_fields = set(parent.fields.values())
    for value in values:
        for field in value.find(Field):
            if field not in possible_fields:
                raise IntegrityError(
                    f"Cannot add {field!r} to projection, "
                    f"it belongs to {field.rel!r}"
                )


class Project(Relation):
    parent: Relation
    values: FrozenDict[str, Annotated[Value, ~InstanceOf(Alias)]]

    def __init__(self, parent, values):
        _check_integrity(parent, values.values())
        super().__init__(parent=parent, values=values)

    @attribute
    def fields(self):
        return self.values


# TODO(kszucs): consider to have a specialization projecting only fields not
# generic value expressions
# class ProjectFields(Relation):
#     parent: Relation
#     fields: FrozenDict[str, Field]

#     @attribute
#     def schema(self):
#         return Schema({f.name: f.dtype for f in self.fields})


# class Join(Relation):
#     left: Relation
#     right: Relation
#     fields: FrozenDict[str, Field]
#     predicates: VarTuple[Value[dt.Boolean]]


class Sort(Relation):
    parent: Relation
    keys: VarTuple[SortKey]

    def __init__(self, parent, keys):
        _check_integrity(parent, keys)
        super().__init__(parent=parent, keys=keys)

    @attribute
    def fields(self):
        return self.parent.fields


class Filter(Relation):
    parent: Relation
    predicates: VarTuple[Value[dt.Boolean]]

    def __init__(self, parent, predicates):
        _check_integrity(parent, predicates)
        super().__init__(parent=parent, predicates=predicates)

    @attribute
    def fields(self):
        return self.parent.fields


# class Aggregate(Relation):
#     parent: Relation
#     groups: FrozenDict[str, Annotated[Column, ~InstanceOf(Alias)]]
#     metrics: FrozenDict[str, Annotated[Scalar, ~InstanceOf(Alias)]]

#     @attribute
#     def schema(self):
#         # schema is consisting both by and metrics, use .from_tuples() to disallow
#         # duplicate names in the schema
#         return Schema.from_tuples([*self.groups.items(), *self.metrics.items()])


class UnboundTable(Relation):
    name: str
    schema: Schema

    @attribute
    def fields(self):
        return {k: Field(self, k) for k in self.schema}


################################ TYPES ################################


class TableExpr(Expr):
    def schema(self):
        return self.op().schema

    def __getattr__(self, key):
        return next(bind(self, key))

    def select(self, *args, **kwargs):
        values = bind(self, (args, kwargs))
        values = unwrap_aliases(values)
        # TODO(kszucs): windowization of reductions should happen here

        rel = self.op()
        if isinstance(rel, Project):
            # subsequent projections, use only the new values
            # rule = p.Field(rel, name) >> Item(rel.values, name)
            # values = {k: v.replace(rule) for k, v in values.items()}
            # this peeling is currently done in bind() directly referencing the
            # parent relation's value
            node = rel.copy(values=values)
        else:
            node = Project(rel, values)

        # node = node.replace(complete_reprojection | subsequent_projections)

        return node.to_expr()

    def where(self, *predicates):
        preds = bind(self, predicates)
        preds = unwrap_aliases(predicates).values()
        # TODO(kszucs): add predicate flattening
        node = Filter(self, preds)
        node = node.replace(subsequent_filters)
        return node.to_expr()

    def order_by(self, *keys):
        keys = bind(self, keys)
        keys = unwrap_aliases(keys).values()
        node = Sort(self, keys)
        node = node.replace(subsequent_sorts)
        return node.to_expr()

    def aggregate(self, groups, metrics):
        groups = bind(self, groups)
        metrics = bind(self, metrics)
        groups = unwrap_aliases(groups)
        metrics = unwrap_aliases(metrics)
        node = Aggregate(self, groups, metrics)
        return node.to_expr()


def bind(table: TableExpr, value: Any) -> ir.Value:
    node = table.op()
    if isinstance(value, ValueExpr):
        yield value
    elif isinstance(value, TableExpr):
        for name in value.schema().keys():
            yield Field(value, name).to_expr()
    elif isinstance(value, str):
        # column peeling / dereferencing
        yield node.fields[value].to_expr().name(value)
    elif isinstance(value, Deferred):
        yield value.resolve(table)
    elif isinstance(value, Selector):
        yield from value.expand(table)
    elif isinstance(value, tuple):
        for v in value:
            yield from bind(table, v)
    elif isinstance(value, dict):
        for k, v in value.items():
            for val in bind(table, v):
                yield val.name(k)
    elif callable(value):
        yield value(table)
    else:
        yield literal(value)


def unwrap_aliases(values):
    result = {}
    for value in values:
        node = value.op()
        if isinstance(node, Alias):
            result[node.name] = node.arg
        else:
            result[node.name] = node
    return result


################################ REWRITES ################################


p = Namespace(pattern, module=__name__)
d = Namespace(deferred, module=__name__)

name = var("name")

x = var("x")
y = var("y")
values = var("values")


@replace(p.Project(y @ p.Relation) & Check(_.schema == y.schema))
def complete_reprojection(_, y):
    # TODO(kszucs): this could be moved to the pattern itself but not sure how
    # to express it, especially in a shorter way then the following check
    for name in _.schema:
        if _.values[name] != Field(y, name):
            return x
    return y


@replace(p.Project(x @ p.Project(y)))
def subsequent_projections(_, x, y):
    rule = p.Field(x, name) >> Item(x.values, name)
    vals = {k: v.replace(rule) for k, v in _.values.items()}
    return Project(y, vals)


@replace(p.Filter(x @ p.Filter))
def subsequent_filters(_, x):
    # this can be easily expressed in simple deferred-like expressions
    return Filter(x.parent, x.predicates + _.predicates)


@replace(p.Sort(x @ p.Sort))
def subsequent_sorts(_, x):
    return Sort(x.parent, x.keys + _.keys)


def relation_of(value):
    def fn(node):
        if isinstance(node, Relation):
            return False, node
        else:
            return True, None

    try:
        return next(traverse(fn, value))
    except StopIteration:
        return None


# POSSIBLE REWRITES:
# 1. Reprojection of the whole relation: t.select(t) >> t
