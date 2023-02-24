from __future__ import annotations

import parsy
import sqlalchemy as sa
from sqlalchemy.ext.compiler import compiles

import ibis.expr.datatypes as dt
from ibis.common.parsing import (
    LANGLE,
    RANGLE,
    spaceless_string,
)


@compiles(sa.BIGINT, "druid")
@compiles(sa.BigInteger, "druid")
def _bigint(element, compiler, **kw):
    return "BIGINT"


@compiles(sa.INTEGER, "druid")
@compiles(sa.Integer, "druid")
def _integer(element, compiler, **kw):
    return "INTEGER"


@compiles(sa.SMALLINT, "druid")
@compiles(sa.SmallInteger, "druid")
def _smallint(element, compiler, **kw):
    return "SMALLINT"


def parse(text: str) -> dt.DataType:
    """Parse a Druid type into an ibis data type."""
    primitive = (
        spaceless_string("string").result(dt.string)
        | spaceless_string("double").result(dt.float64)
        | spaceless_string("float").result(dt.float32)
        | spaceless_string("long").result(dt.int64)
        | spaceless_string("json").result(dt.json)
    )

    ty = parsy.forward_declaration()

    json = spaceless_string("complex").then(LANGLE).then(ty).skip(RANGLE)
    array = spaceless_string("array").then(LANGLE).then(ty.map(dt.Array)).skip(RANGLE)

    ty.become(primitive | array | json)
    return ty.parse(text)
