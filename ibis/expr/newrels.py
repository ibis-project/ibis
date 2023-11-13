from __future__ import annotations

import types
from abc import abstractmethod
from functools import wraps
from typing import Annotated, Any, Optional

import ibis.expr.datashape as ds
import ibis.expr.datatypes as dt
import ibis.expr.types as ir
from ibis.common.annotations import attribute
from ibis.common.collections import FrozenDict  # noqa: TCH001
from ibis.common.deferred import Deferred, Item, deferred, var
from ibis.common.exceptions import IntegrityError
from ibis.common.grounds import Concrete
from ibis.common.patterns import Check, InstanceOf, _, pattern, replace
from ibis.common.typing import Coercible, VarTuple
from ibis.expr.operations import Alias, Node, SortKey, Value
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

    @property
    @abstractmethod
    def fields(self):
        ...

    @property
    @abstractmethod
    def schema(self):
        ...

    def to_expr(self):
        return TableExpr(self)


class Field(Value):
    rel: Relation
    name: str

    shape = ds.columnar

    @property
    def dtype(self):
        return self.rel.schema[self.name]


def _check_integrity(parent, values):
    possible_fields = set(parent.fields.values())
    for value in values:
        for field in value.find(Field, filter=Value):
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

    @property
    def fields(self):
        return {k: Field(self, k) for k in self.values}

    @property
    def schema(self):
        return Schema({k: v.dtype for k, v in self.values.items()})


# TODO(kszucs): consider to have a specialization projecting only fields not
# generic value expressions
# class ProjectFields(Relation):
#     parent: Relation
#     fields: FrozenDict[str, Field]

#     @attribute
#     def schema(self):
#         return Schema({f.name: f.dtype for f in self.fields})


class Join(Relation):
    left: Relation
    right: Relation
    fields: FrozenDict[str, Field]
    predicates: VarTuple[Value[dt.Boolean]]
    how: str = "inner"

    def __init__(self, left, right, fields, predicates, how):
        # _check_integrity(left, predicates)
        # _check_integrity(right, predicates)
        super().__init__(
            left=left, right=right, fields=fields, predicates=predicates, how=how
        )

    @property
    def schema(self):
        return Schema({k: v.dtype for k, v in self.fields.items()})


class Sort(Relation):
    parent: Relation
    keys: VarTuple[SortKey]

    def __init__(self, parent, keys):
        _check_integrity(parent, keys)
        super().__init__(parent=parent, keys=keys)

    @property
    def fields(self):
        return self.parent.fields

    @property
    def schema(self):
        return self.parent.schema


class Filter(Relation):
    parent: Relation
    predicates: VarTuple[Value[dt.Boolean]]

    def __init__(self, parent, predicates):
        # TODO(kszucs): use toolz.unique(predicates) to remove duplicates
        _check_integrity(parent, predicates)
        super().__init__(parent=parent, predicates=predicates)

    @property
    def fields(self):
        return self.parent.fields

    @property
    def schema(self):
        return self.parent.schema


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

    @property
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
        node = Project(self, values)
        node = node.replace(complete_reprojection | subsequent_projects)
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

    def join(self, right, predicates, how="inner"):
        return JoinChain(self).join(right, predicates, how)

    def aggregate(self, groups, metrics):
        groups = bind(self, groups)
        metrics = bind(self, metrics)
        groups = unwrap_aliases(groups)
        metrics = unwrap_aliases(metrics)
        node = Aggregate(self, groups, metrics)
        return node.to_expr()


class JoinLink(Concrete):
    how: str = "inner"
    right: Relation
    predicates: VarTuple[Value[dt.Boolean]]


class JoinChain(Concrete):
    start: Relation
    links: VarTuple[JoinLink] = ()

    def join(self, right, predicates, how="inner"):
        # do the predicate coercion and binding here
        link = JoinLink(how=how, right=right, predicates=predicates)
        return self.copy(links=self.links + (link,))

    def select(self, *args, **kwargs):
        # do the fields projection here
        values = bind(self, (args, kwargs))
        values = unwrap_aliases(values)

        # TODO(kszucs): go over the values and pull out the fields only, until
        # that just raise if the value is a computed expression
        for value in values.values():
            if not isinstance(value, Field):
                raise TypeError("Only fields can be selected in a join")

        return self.finish(values)

    def guess(self):
        # TODO(kszucs): more advanced heuristics can be applied here
        # try to figure out the join intent here
        fields = self.start.fields
        for link in self.links:
            if fields.keys() & link.right.fields.keys():
                # overlapping fields
                raise IntegrityError("Overlapping fields in join")
            fields.update(link.right.fields)
        return fields

    def finish(self, fields=None):
        if fields is None:
            fields = self.guess()

        # build the join operation from the join chain
        left = self.start
        for link in self.links:
            # pick only the fields relevant at this level of the join since fields
            # can contain references from all of the links but we are only interested
            # in the ones belonging to the current link (part of left and link.right)
            # TODO(kszucs): must put fields being part of the predicates here as well
            possible_fields = {*left.fields.values(), *link.right.fields.values()}
            selected_fields = {k: v for k, v in fields.items() if v in possible_fields}

            left = Join(
                how=link.how,
                left=left,
                right=link.right,
                predicates=link.predicates,
                fields=selected_fields,
            )
        return left.to_expr()

    def __dir__(self):
        return dir(TableExpr)

    def __getattr__(self, key):
        method = getattr(TableExpr, key)

        @wraps(method)
        def wrapper(*args, **kwargs):
            return method(self.finish(), *args, **kwargs)

        return wrapper


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


@replace(p.Project(y @ p.Project))
def subsequent_projects(_, y):
    rule = p.Field(y, name) >> Item(y.values, name)
    values = {k: v.replace(rule) for k, v in _.values.items()}
    return Project(y.parent, values)


@replace(p.Filter(y @ p.Filter))
def subsequent_filters(_, y):
    return Filter(y.parent, y.predicates + _.predicates)


@replace(p.Sort(y @ p.Sort))
def subsequent_sorts(_, y):
    return Sort(y.parent, y.keys + _.keys)


# TODO(kszucs): support t.select(*t) syntax by implementing TableExpr.__iter__()
