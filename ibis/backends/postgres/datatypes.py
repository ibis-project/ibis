from __future__ import annotations

import ibis.backends.duckdb.datatypes as ddb
import ibis.expr.datatypes as dt


def _get_type(typestr: str) -> dt.DataType:
    try:
        return _type_mapping[typestr]
    except KeyError:
        return ddb.parse(typestr)


_type_mapping = {
    "boolean": dt.bool,
    "boolean[]": dt.Array(dt.bool),
    "bytea": dt.binary,
    "bytea[]": dt.Array(dt.binary),
    "character(1)": dt.string,
    "character(1)[]": dt.Array(dt.string),
    "bigint": dt.int64,
    "bigint[]": dt.Array(dt.int64),
    "smallint": dt.int16,
    "smallint[]": dt.Array(dt.int16),
    "integer": dt.int32,
    "integer[]": dt.Array(dt.int32),
    "text": dt.string,
    "text[]": dt.Array(dt.string),
    "json": dt.json,
    "json[]": dt.Array(dt.json),
    "point": dt.point,
    "point[]": dt.Array(dt.point),
    "polygon": dt.polygon,
    "polygon[]": dt.Array(dt.polygon),
    "line": dt.linestring,
    "line[]": dt.Array(dt.linestring),
    "real": dt.float32,
    "real[]": dt.Array(dt.float32),
    "double precision": dt.float64,
    "double precision[]": dt.Array(dt.float64),
    "macaddr8": dt.macaddr,
    "macaddr8[]": dt.Array(dt.macaddr),
    "macaddr": dt.macaddr,
    "macaddr[]": dt.Array(dt.macaddr),
    "inet": dt.inet,
    "inet[]": dt.Array(dt.inet),
    "character": dt.string,
    "character[]": dt.Array(dt.string),
    "character varying": dt.string,
    "character varying[]": dt.Array(dt.string),
    "date": dt.date,
    "date[]": dt.Array(dt.date),
    "time without time zone": dt.time,
    "time without time zone[]": dt.Array(dt.time),
    "timestamp without time zone": dt.timestamp,
    "timestamp without time zone[]": dt.Array(dt.timestamp),
    "timestamp with time zone": dt.Timestamp("UTC"),
    "timestamp with time zone[]": dt.Array(dt.Timestamp("UTC")),
    "interval": dt.interval,
    "interval[]": dt.Array(dt.interval),
    # NB: this isn"t correct, but we try not to fail
    "time with time zone": "time",
    "numeric": dt.decimal,
    "numeric[]": dt.Array(dt.decimal),
    "uuid": dt.uuid,
    "uuid[]": dt.Array(dt.uuid),
    "jsonb": dt.json,
    "jsonb[]": dt.Array(dt.json),
    "geometry": dt.geometry,
    "geometry[]": dt.Array(dt.geometry),
    "geography": dt.geography,
    "geography[]": dt.Array(dt.geography),
}
