from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import sqlglot as sg

import ibis.expr.datatypes as dt

if TYPE_CHECKING:
    from collections.abc import Iterable

fully_qualified_re = re.compile(r"(.*)\.(?:`(.*)`|(.*))")


# TODO(kszucs): the following classes should produce SQLGlot expressions


def is_fully_qualified(x):
    return bool(fully_qualified_re.search(x))


def _is_quoted(x):
    regex = re.compile(r"(?:`(.*)`|(.*))")
    quoted, _ = regex.match(x).groups()
    return quoted is not None


class Base(ABC):
    @property
    @abstractmethod
    def dialect(self): ...

    @abstractmethod
    def compile(self): ...

    def quote(self, ident: str) -> str:
        return sg.to_identifier(ident, quoted=True).sql(dialect=self.dialect)

    def format_literal(self, value: Any, type: dt.DataType | str) -> str:
        type = dt.dtype(type)
        if value is None:
            return "NULL"
        if type.is_string():
            return f"'{value}'"
        return str(value)

    def scoped_name(
        self, obj_name: str, database: str | None = None, catalog: str | None = None
    ) -> str:
        if is_fully_qualified(obj_name):
            return obj_name
        if _is_quoted(obj_name):
            obj_name = obj_name[1:-1]
        return sg.table(obj_name, db=database, catalog=catalog, quoted=True).sql(
            dialect=self.dialect
        )

    @abstractmethod
    def format_dtype(self, dtype): ...

    def format_schema(self, schema):
        elements = [
            f"{self.quote(name)} {self.format_dtype(t)}"
            for name, t in zip(schema.names, schema.types)
        ]
        return "({})".format(",\n ".join(elements))

    def format_partition(
        self,
        partition: dict[str, Any] | Iterable[str],
        partition_schema: dict[str, dt.DataType],
    ) -> str:
        def _format_partition_kv(k: str, v: Any, type: dt.DataType) -> str:
            return f"{self.quote(k)}={self.format_literal(v, type)}"

        tokens = []
        if isinstance(partition, dict):
            for name, dtype in partition_schema.items():
                if name in partition:
                    tok = _format_partition_kv(name, partition[name], dtype)
                else:
                    # dynamic partitioning
                    tok = self.quote(name)
                tokens.append(tok)
        else:
            for name, value in zip(partition_schema, partition):
                tok = _format_partition_kv(name, value, partition_schema[name])
                tokens.append(tok)

        return "PARTITION ({})".format(", ".join(tokens))


class DML(Base):
    pass


class DDL(Base):
    pass


class CreateDDL(DDL):
    def _if_exists(self):
        return "IF NOT EXISTS " if self.can_exist else ""


class DropObject(DDL):
    def __init__(self, must_exist=True):
        self.must_exist = must_exist

    def compile(self):
        if_exists = "" if self.must_exist else "IF EXISTS "
        object_name = self._object_name()
        return f"DROP {self._object_type} {if_exists}{object_name}"


class DropFunction(DropObject):
    def __init__(self, name, inputs, must_exist=True, aggregate=False, database=None):
        super().__init__(must_exist=must_exist)
        self.name = name
        self.inputs = tuple(map(dt.dtype, inputs))
        self.must_exist = must_exist
        self.aggregate = aggregate
        self.database = database

    def _object_name(self):
        return self.name

    def compile(self):
        tokens = ["DROP"]
        if self.aggregate:
            tokens.append("AGGREGATE")
        tokens.append("FUNCTION")
        if not self.must_exist:
            tokens.append("IF EXISTS")

        tokens.append(self._impala_signature())
        return " ".join(tokens)
