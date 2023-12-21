from __future__ import annotations

import itertools
import typing
from abc import abstractmethod
from typing import Annotated, Any, Literal, Optional, TypeVar

from public import public

import ibis.expr.datashape as ds
import ibis.expr.datatypes as dt
import ibis.expr.rules as rlz
from ibis.common.annotations import attribute
from ibis.common.collections import FrozenDict
from ibis.common.exceptions import IbisTypeError, IntegrityError, RelationError
from ibis.common.grounds import Concrete
from ibis.common.patterns import Between, InstanceOf
from ibis.common.typing import Coercible, VarTuple
from ibis.expr.operations.core import Alias, Column, Node, Scalar, Value
from ibis.expr.operations.sortkeys import SortKey  # noqa: TCH001
from ibis.expr.schema import Schema
from ibis.formats import TableProxy  # noqa: TCH001
from ibis.util import gen_name

T = TypeVar("T")

Unaliased = Annotated[T, ~InstanceOf(Alias)]


@public
class Relation(Node, Coercible):
    @classmethod
    def __coerce__(cls, value):
        from ibis.expr.types import TableExpr

        if isinstance(value, Relation):
            return value
        elif isinstance(value, TableExpr):
            return value.op()
        else:
            raise TypeError(f"Cannot coerce {value!r} to a Relation")

    @property
    @abstractmethod
    def values(self) -> FrozenDict[str, Value]:
        """A mapping of column names to expressions which build up the relation.

        This attribute is heavily used in rewrites as well as during field
        dereferencing in the API layer. The returned expressions must only
        originate from parent relations, depending on the relation type.
        """

    @property
    @abstractmethod
    def schema(self) -> Schema:
        """The schema of the relation.

        All relations must have a well-defined schema.
        """
        ...

    @property
    def fields(self) -> FrozenDict[str, Column]:
        """A mapping of column names to fields of the relation.

        This calculated property shouldn't be overridden in subclasses since it
        is mostly used for convenience.
        """
        return FrozenDict({k: Field(self, k) for k in self.schema})

    def to_expr(self):
        from ibis.expr.types import TableExpr

        return TableExpr(self)


@public
class Field(Value):
    rel: Relation
    name: str

    shape = ds.columnar

    def __init__(self, rel, name):
        if name not in rel.schema:
            columns_formatted = ", ".join(map(repr, rel.schema.names))
            raise IbisTypeError(
                f"Column {name!r} is not found in table. "
                f"Existing columns: {columns_formatted}."
            )
        super().__init__(rel=rel, name=name)

    @attribute
    def dtype(self):
        return self.rel.schema[self.name]

    @attribute
    def relations(self):
        return frozenset({self.rel})


@public
class Subquery(Value):
    rel: Relation
    shape = ds.columnar

    def __init__(self, rel, **kwargs):
        if len(rel.schema) != 1:
            raise IntegrityError(
                f"Subquery must have exactly one column, got {len(rel.schema)}"
            )
        super().__init__(rel=rel, **kwargs)

    @attribute
    def name(self):
        return self.rel.schema.names[0]

    @attribute
    def value(self):
        return self.rel.values[self.name]

    @attribute
    def relations(self):
        return frozenset()

    @property
    def dtype(self):
        return self.value.dtype


@public
class ScalarSubquery(Subquery):
    def __init__(self, rel):
        from ibis.expr.rewrites import ReductionValue

        super().__init__(rel=rel)
        if not self.value.find(ReductionValue, filter=Value):
            raise IntegrityError(
                f"Subquery {self.value!r} is not scalar, it must be turned into a scalar subquery first"
            )


@public
class ExistsSubquery(Subquery):
    dtype = dt.boolean


@public
class InSubquery(Subquery):
    needle: Value
    dtype = dt.boolean

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not rlz.comparable(self.value, self.needle):
            raise IntegrityError(
                f"Subquery {self.needle!r} is not comparable to {self.value!r}"
            )

    @attribute
    def relations(self):
        return self.needle.relations


def _check_integrity(values, allowed_parents):
    for value in values:
        for rel in value.relations:
            if rel not in allowed_parents:
                raise IntegrityError(
                    f"Cannot add {value!r} to projection, they belong to another relation"
                )


@public
class Project(Relation):
    parent: Relation
    values: FrozenDict[str, Unaliased[Value]]

    def __init__(self, parent, values):
        _check_integrity(values.values(), {parent})
        super().__init__(parent=parent, values=values)

    @attribute
    def schema(self):
        return Schema({k: v.dtype for k, v in self.values.items()})


class Simple(Relation):
    parent: Relation

    @attribute
    def values(self):
        return self.parent.fields

    @attribute
    def schema(self):
        return self.parent.schema


@public
class SelfReference(Simple):
    _uid_counter = itertools.count()

    identifier: Optional[int] = None

    def __init__(self, parent, identifier):
        if identifier is None:
            identifier = next(self._uid_counter)
        super().__init__(parent=parent, identifier=identifier)

    @attribute
    def name(self) -> str:
        if (name := getattr(self.parent, "name", None)) is not None:
            return f"{name}_ref"
        return gen_name("self_ref")


JoinKind = Literal[
    "inner",
    "left",
    "right",
    "outer",
    "asof",
    "semi",
    "anti",
    "any_inner",
    "any_left",
    "cross",
]


@public
class JoinLink(Node):
    how: JoinKind
    table: SelfReference
    predicates: VarTuple[Value[dt.Boolean]]


@public
class JoinChain(Relation):
    first: SelfReference
    rest: VarTuple[JoinLink]
    values: FrozenDict[str, Unaliased[Value]]

    def __init__(self, first, rest, values):
        allowed_parents = {first}
        for join in rest:
            allowed_parents.add(join.table)
            _check_integrity(join.predicates, allowed_parents)
        _check_integrity(values.values(), allowed_parents)
        super().__init__(first=first, rest=rest, values=values)

    @attribute
    def schema(self):
        return Schema({k: v.dtype.copy(nullable=True) for k, v in self.values.items()})

    def to_expr(self):
        import ibis.expr.types as ir

        return ir.JoinExpr(self)


@public
class Sort(Simple):
    keys: VarTuple[SortKey]

    def __init__(self, parent, keys):
        _check_integrity(keys, {parent})
        super().__init__(parent=parent, keys=keys)


@public
class Filter(Simple):
    predicates: VarTuple[Value[dt.Boolean]]

    def __init__(self, parent, predicates):
        from ibis.expr.rewrites import ReductionValue

        for pred in predicates:
            if pred.find(ReductionValue, filter=Value):
                raise IntegrityError(
                    f"Cannot add {pred!r} to filter, it is a reduction"
                )
            if pred.relations and parent not in pred.relations:
                raise IntegrityError(
                    f"Cannot add {pred!r} to filter, they belong to another relation"
                )
        super().__init__(parent=parent, predicates=predicates)


@public
class Limit(Simple):
    n: typing.Union[int, Scalar[dt.Integer], None] = None
    offset: typing.Union[int, Scalar[dt.Integer]] = 0


@public
class Aggregate(Relation):
    parent: Relation
    groups: FrozenDict[str, Unaliased[Column]]
    metrics: FrozenDict[str, Unaliased[Scalar]]

    def __init__(self, parent, groups, metrics):
        _check_integrity(groups.values(), {parent})
        _check_integrity(metrics.values(), {parent})
        if duplicates := groups.keys() & metrics.keys():
            raise RelationError(
                f"Cannot add {duplicates} to aggregate, they are already in the groupby"
            )
        super().__init__(parent=parent, groups=groups, metrics=metrics)

    @attribute
    def values(self):
        return FrozenDict({**self.groups, **self.metrics})

    @attribute
    def schema(self):
        return Schema({k: v.dtype for k, v in self.values.items()})


@public
class Set(Relation):
    left: Relation
    right: Relation
    distinct: bool = False

    def __init__(self, left, right, **kwargs):
        # convert to dictionary first, to get key-unordered comparison semantics
        if dict(left.schema) != dict(right.schema):
            raise RelationError("Table schemas must be equal for set operations")
        elif left.schema.names != right.schema.names:
            # rewrite so that both sides have the columns in the same order making it
            # easier for the backends to implement set operations
            cols = {name: Field(right, name) for name in left.schema.names}
            right = Project(right, cols)
        super().__init__(left=left, right=right, **kwargs)

    @attribute
    def values(self):
        return FrozenDict()

    @attribute
    def schema(self):
        return self.left.schema


@public
class Union(Set):
    pass


@public
class Intersection(Set):
    pass


@public
class Difference(Set):
    pass


@public
class PhysicalTable(Relation):
    name: str

    @attribute
    def values(self):
        return FrozenDict()


@public
class UnboundTable(PhysicalTable):
    schema: Schema


@public
class Namespace(Concrete):
    database: Optional[str] = None
    schema: Optional[str] = None


@public
class DatabaseTable(PhysicalTable):
    schema: Schema
    source: Any
    namespace: Namespace = Namespace()


@public
class InMemoryTable(PhysicalTable):
    schema: Schema
    data: TableProxy


@public
class SQLQueryResult(Relation):
    """A table sourced from the result set of a select query."""

    query: str
    schema: Schema
    source: Any
    values = FrozenDict()


@public
class SQLStringView(PhysicalTable):
    """A view created from a SQL string."""

    child: Relation
    query: str

    @attribute
    def schema(self):
        # TODO(kszucs): avoid converting to expression
        backend = self.child.to_expr()._find_backend()
        return backend._get_schema_using_query(self.query)


@public
class DummyTable(Relation):
    values: FrozenDict[str, Value]

    @attribute
    def schema(self):
        return Schema({k: v.dtype for k, v in self.values.items()})


@public
class FillNa(Simple):
    """Fill null values in the table."""

    replacements: typing.Union[Value[dt.Numeric | dt.String], FrozenDict[str, Any]]


@public
class DropNa(Simple):
    """Drop null values in the table."""

    how: typing.Literal["any", "all"]
    subset: Optional[VarTuple[Column]] = None


@public
class Sample(Simple):
    """Sample performs random sampling of records in a table."""

    fraction: Annotated[float, Between(0, 1)]
    method: typing.Literal["row", "block"]
    seed: typing.Union[int, None] = None


@public
class Distinct(Simple):
    """Distinct is a table-level unique-ing operation."""


# TODO(kszucs): support t.select(*t) syntax by implementing TableExpr.__iter__()
