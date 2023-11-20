from __future__ import annotations

# need a parallel Expression and Operation class hierarchy to decompose ops.Selection
# into proper relational algebra operations
################################ OPERATIONS ################################
import itertools
import typing
from abc import abstractmethod
from typing import Annotated, Any, Literal, Optional

from public import public

import ibis.expr.datashape as ds
import ibis.expr.datatypes as dt
from ibis.common.annotations import attribute
from ibis.common.bases import Immutable
from ibis.common.collections import FrozenDict
from ibis.common.deferred import Item, deferred, var
from ibis.common.exceptions import IbisTypeError, IntegrityError, RelationError
from ibis.common.grounds import Concrete
from ibis.common.patterns import Between, Check, In, InstanceOf, _, pattern, replace
from ibis.common.typing import Coercible, VarTuple
from ibis.expr.operations.core import Alias, Column, Node, Scalar, Value
from ibis.expr.operations.sortkeys import SortKey
from ibis.expr.schema import Schema
from ibis.util import Namespace, gen_name, indent

p = Namespace(pattern, module=__name__)
d = Namespace(deferred, module=__name__)

################################## InMemoryTable ##################################


class TableProxy(Immutable):
    __slots__ = ("_data", "_hash")
    _data: Any
    _hash: int

    def __init__(self, data) -> None:
        object.__setattr__(self, "_data", data)
        object.__setattr__(self, "_hash", hash((type(data), id(data))))

    def __hash__(self) -> int:
        return self._hash

    def __repr__(self) -> str:
        data_repr = indent(repr(self._data), spaces=2)
        return f"{self.__class__.__name__}:\n{data_repr}"

    @abstractmethod
    def to_frame(self) -> pd.DataFrame:  # pragma: no cover
        """Convert this input to a pandas DataFrame."""

    @abstractmethod
    def to_pyarrow(self, schema: Schema) -> pa.Table:  # pragma: no cover
        """Convert this input to a PyArrow Table."""

    def to_pyarrow_bytes(self, schema: Schema) -> bytes:
        import pyarrow as pa
        import pyarrow_hotfix  # noqa: F401

        data = self.to_pyarrow(schema=schema)
        out = pa.BufferOutputStream()
        with pa.RecordBatchFileWriter(out, data.schema) as writer:
            writer.write(data)
        return out.getvalue()

    def __len__(self) -> int:
        return len(self._data)


class PyArrowTableProxy(TableProxy):
    __slots__ = ()

    def to_frame(self):
        return self._data.to_pandas()

    def to_pyarrow(self, schema: Schema) -> pa.Table:
        return self._data


class PandasDataFrameProxy(TableProxy):
    __slots__ = ()

    def to_frame(self) -> pd.DataFrame:
        return self._data

    def to_pyarrow(self, schema: Schema) -> pa.Table:
        import pyarrow as pa
        import pyarrow_hotfix  # noqa: F401

        from ibis.formats.pyarrow import PyArrowSchema

        return pa.Table.from_pandas(self._data, schema=PyArrowSchema.from_ibis(schema))


############################################################################


@public
class Relation(Node, Coercible):
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._counter = itertools.count()

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
    def fields(self):
        # used for dereferencing fields/values to the lowest relation in the
        # chain this should return references to parent relation's fields
        ...

    @property
    @abstractmethod
    def schema(self):
        ...

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
            # TODO(kszucs): should raise a more specific error type
            raise IbisTypeError(
                f"Column {name!r} is not found in table. "
                f"Existing columns: {columns_formatted}."
            )
        super().__init__(rel=rel, name=name)

    @attribute
    def dtype(self):
        return self.rel.schema[self.name]


@public
class ForeignField(Value):
    rel: Relation
    name: str

    shape = ds.columnar

    @attribute
    def dtype(self):
        return self.rel.schema[self.name]


def _check_integrity(values, allowed_parents):
    foreign_field = p.Field(~In(allowed_parents))
    for value in values:
        if disallowed := value.match(foreign_field, filter=Value):
            raise IntegrityError(
                f"Cannot add {disallowed!r} to projection, they belong to another relation"
            )
        # egyebkent csak scalar lehet (e.g. scalar subquery or a value based on literals)


@public
class Project(Relation):
    parent: Relation
    values: FrozenDict[str, Annotated[Value, ~InstanceOf(Alias)]]

    def __init__(self, parent, values):
        _check_integrity(values.values(), {parent})
        # TODO(kszucs): raise if values depending on foreign fields are not scalar shaped
        # TODO(kszucs): column-like values dependent on foreign fields are allowed in filter predicates only
        # TODO(kszucs): additional integrity check can be done for correlated subqueryies:
        # 1. locate the values with foreign fields in this projection
        # 2. locate the foreign fields in the relations of the values above
        # 3. assert that the relation of those foreign fields is `parent`
        # this way we can ensure that the foreign fields are not referencing relations
        # foreign to the currently constructed one, but there are just references
        # back and forth

        # TODO(kszucs): move this to the integrity checker?
        for v in values.values():
            if v.find(ForeignField) and not v.shape.is_scalar():
                raise IntegrityError(
                    f"Cannot add foreign value {v!r} to projection, it is not scalar shaped"
                )

        super().__init__(parent=parent, values=values)

    @attribute
    def fields(self):
        return self.values

    @attribute
    def schema(self):
        return Schema({k: v.dtype for k, v in self.values.items()})


# TODO(kszucs): Subquery(value, outer_relation)


@public
class JoinLink(Node):
    how: Literal["inner", "left", "right", "outer", "asof", "semi", "cross"]
    table: Relation
    predicates: VarTuple[Value[dt.Boolean]]


@public
class JoinChain(Relation):
    first: Relation
    rest: VarTuple[JoinLink]
    fields: FrozenDict[str, Field]

    def __init__(self, first, rest, fields):
        allowed_parents = {first}
        for join in rest:
            allowed_parents.add(join.table)
            _check_integrity(join.predicates, allowed_parents)
        _check_integrity(fields.values(), allowed_parents)
        super().__init__(first=first, rest=rest, fields=fields)

    @attribute
    def schema(self):
        return Schema({k: v.dtype for k, v in self.fields.items()})


@public
class Sort(Relation):
    parent: Relation
    keys: VarTuple[SortKey]

    def __init__(self, parent, keys):
        _check_integrity(keys, {parent})
        super().__init__(parent=parent, keys=keys)

    @attribute
    def fields(self):
        return FrozenDict({k: Field(self.parent, k) for k in self.parent.schema})

    @attribute
    def schema(self):
        return self.parent.schema


@public
class Filter(Relation):
    parent: Relation
    predicates: VarTuple[Value[dt.Boolean]]

    def __init__(self, parent, predicates):
        # TODO(kszucs): use toolz.unique(predicates) to remove duplicates
        _check_integrity(predicates, {parent})
        super().__init__(parent=parent, predicates=predicates)

    @attribute
    def fields(self):
        return FrozenDict({k: Field(self.parent, k) for k in self.parent.schema})

    @attribute
    def schema(self):
        return self.parent.schema


@public
class Limit(Relation):
    parent: Relation
    n: typing.Union[int, Scalar[dt.Integer], None] = None
    offset: typing.Union[int, Scalar[dt.Integer]] = 0

    @attribute
    def fields(self):
        return FrozenDict({k: Field(self.parent, k) for k in self.parent.schema})

    @attribute
    def schema(self):
        return self.parent.schema


@public
class Aggregate(Relation):
    parent: Relation
    groups: FrozenDict[str, Annotated[Column, ~InstanceOf(Alias)]]
    metrics: FrozenDict[str, Annotated[Scalar, ~InstanceOf(Alias)]]

    def __init__(self, parent, groups, metrics):
        _check_integrity(groups.values(), {parent})
        _check_integrity(metrics.values(), {parent})
        super().__init__(parent=parent, groups=groups, metrics=metrics)

    @attribute
    def fields(self):
        return FrozenDict({**self.groups, **self.metrics})

    @attribute
    def schema(self):
        # schema is consisting both by and metrics, use .from_tuples() to disallow
        # duplicate names in the schema
        groups = {k: v.dtype for k, v in self.groups.items()}
        metrics = {k: v.dtype for k, v in self.metrics.items()}
        return Schema.from_tuples([*groups.items(), *metrics.items()])


@public
class Set(Relation):
    left: Relation
    right: Relation
    distinct: bool = False

    def __init__(self, left, right, **kwargs):
        # convert to dictionary first, to get key-unordered comparison
        # semantics
        if dict(left.schema) != dict(right.schema):
            raise RelationError("Table schemas must be equal for set operations")
        elif left.schema.names != right.schema.names:
            # rewrite so that both sides have the columns in the same order making it
            # easier for the backends to implement set operations
            cols = [Field(right, name) for name in left.schema.names]
            right = Project(right, cols)
        super().__init__(left=left, right=right, **kwargs)

    @attribute
    def fields(self):
        return {}  # FrozenDict({k: Field(self.left, k) for k in self.left.schema})

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


# TODO(kszucs): call it Source or Table?
@public
class PhysicalTable(Relation):
    name: str
    schema: Schema
    fields = FrozenDict()


@public
class UnboundTable(PhysicalTable):
    pass


@public
class Namespace(Concrete):
    database: Optional[str] = None
    schema: Optional[str] = None


@public
class DatabaseTable(PhysicalTable):
    source: Any
    namespace: Namespace = Namespace()


@public
class InMemoryTable(PhysicalTable):
    name: str
    schema: Schema
    data: TableProxy


@public
class SQLQueryResult(Relation):
    """A table sourced from the result set of a select query."""

    query: str
    schema: Schema
    source: Any


@public
class DummyTable(Relation):
    # TODO(kszucs): verify that it has at least one element: Length(at_least=1)
    values: VarTuple[Value[dt.Any, ds.Scalar]]

    @attribute
    def fields(self):
        return self.values

    @attribute
    def schema(self):
        return Schema({op.name: op.dtype for op in self.values})


@public
class SelfReference(Relation):
    parent: Relation

    @attribute
    def name(self) -> str:
        if (name := getattr(self.parent, "name", None)) is not None:
            return f"{name}_ref"
        return gen_name("self_ref")

    @attribute
    def fields(self):
        return {}

    @attribute
    def schema(self):
        return self.parent.schema


@public
class FillNa(Relation):
    """Fill null values in the table."""

    parent: Relation
    replacements: typing.Union[Value[dt.Numeric | dt.String], FrozenDict[str, Any]]

    @attribute
    def fields(self):
        return FrozenDict({k: Field(self.parent, k) for k in self.parent.schema})

    @attribute
    def schema(self):
        return self.parent.schema


@public
class DropNa(Relation):
    """Drop null values in the table."""

    parent: Relation
    how: typing.Literal["any", "all"]
    subset: Optional[VarTuple[Column[dt.Any]]] = None

    @attribute
    def fields(self):
        return FrozenDict({k: Field(self.parent, k) for k in self.parent.schema})

    @attribute
    def schema(self):
        return self.parent.schema


@public
class Sample(Relation):
    """Sample performs random sampling of records in a table."""

    parent: Relation
    fraction: Annotated[float, Between(0, 1)]
    method: typing.Literal["row", "block"]
    seed: typing.Union[int, None] = None

    @attribute
    def fields(self):
        return FrozenDict({k: Field(self.parent, k) for k in self.parent.schema})

    @attribute
    def schema(self):
        return self.parent.schema


@public
class Distinct(Relation):
    """Distinct is a table-level unique-ing operation.

    In SQL, you might have:

    SELECT DISTINCT foo
    FROM table

    SELECT DISTINCT foo, bar
    FROM table
    """

    parent: Relation

    @attribute
    def fields(self):
        return FrozenDict({k: Field(self.parent, k) for k in self.parent.schema})

    @attribute
    def schema(self):
        return self.parent.schema


# class Subquery(Relation):
#     rel: Relation

#     @property
#     def schema(self):
#         return self.rel.schema

#     @property
#     def fields(self):
#         return self.rel.fields


################################ TYPES ################################


# class TableExpr(Expr):
#     def schema(self):
#         return self.op().schema

#     # def __getattr__(self, key):
#     #     return next(bind(self, key))


@public
def table(name, schema):
    return UnboundTable(name, schema).to_expr()


# TODO(kszucs): cover it with tests


################################ REWRITES ################################

name = var("name")

y = var("y")
values = var("values")


@replace(p.Project(y @ p.Relation) & Check(_.schema == y.schema))
def complete_reprojection(_, y):
    # TODO(kszucs): this could be moved to the pattern itself but not sure how
    # to express it, especially in a shorter way then the following check
    for name in _.schema:
        if _.values[name] != Field(y, name):
            return _
    return y


@replace(p.Project(y @ p.Project))
def subsequent_projects(_, y):
    rule = p.Field(y, name) >> Item(y.values, name)
    values = {k: v.replace(rule) for k, v in _.values.items()}
    return Project(y.parent, values)


@replace(p.Filter(y @ p.Filter))
def subsequent_filters(_, y):
    rule = p.Field(y, name) >> d.Field(y.parent, name)
    preds = tuple(v.replace(rule) for v in _.predicates)
    return Filter(y.parent, y.predicates + preds)


# TODO(kszucs): this may work if the sort keys are not overlapping, need to revisit
@replace(p.Sort(y @ p.Sort))
def subsequent_sorts(_, y):
    return Sort(y.parent, y.keys + _.keys)


# TODO(kszucs): support t.select(*t) syntax by implementing TableExpr.__iter__()


# subqueries:
# 1. reduction passed to .filter() should be turned into a subquery
# 2. reduction passed to .select() with a foreign table should be turned into a subquery
