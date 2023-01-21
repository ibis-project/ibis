import pytest
from pytest import param

import ibis.expr.datatypes as dt
from ibis.backends.duckdb.datatypes import parse, parse_type

EXPECTED_SCHEMA = dict(
    a=dt.int64,
    b=dt.int64,
    c=dt.int64,
    d=dt.boolean,
    e=dt.boolean,
    f=dt.boolean,
    g=dt.binary,
    h=dt.binary,
    i=dt.binary,
    j=dt.binary,
    k=dt.date,
    l=dt.float64,
    m=dt.float64,
    n=dt.Decimal(18, 3),
    o=dt.Decimal(18, 3),
    p=dt.Decimal(10, 3),
    q=dt.int32,
    r=dt.int32,
    s=dt.int32,
    t=dt.int32,
    u=dt.interval,
    v=dt.float32,
    w=dt.float32,
    x=dt.float32,
    y=dt.int16,
    z=dt.int16,
    A=dt.int16,
    B=dt.time,
    C=dt.Timestamp('UTC'),
    D=dt.Timestamp('UTC'),
    E=dt.int8,
    F=dt.int8,
    G=dt.uint64,
    H=dt.uint32,
    I=dt.uint16,
    J=dt.uint8,
    K=dt.uuid,
    L=dt.string,
    M=dt.string,
    N=dt.string,
    O=dt.string,
    P=dt.string,
    Q=dt.Array(dt.int32),
    R=dt.Map(dt.string, dt.int64),
    S=dt.Struct(
        dict(
            a=dt.int32,
            b=dt.string,
            c=dt.Array(dt.Map(dt.string, dt.Array(dt.float64))),
        )
    ),
    T=dt.Array(dt.Array(dt.int32)),
    U=dt.Array(dt.Array(dt.int32)),
    V=dt.Timestamp("UTC"),
    W=dt.Timestamp("UTC"),
    X=dt.Timestamp("UTC"),
    Y=dt.Timestamp("UTC"),
    Z=dt.json,
)


@pytest.mark.parametrize(
    ("column", "type"),
    [
        param(colname, type, id=type.lower())
        for colname, type in [
            ("a", "BIGINT"),
            ("b", "INT8"),
            ("c", "LONG"),
            ("d", "BOOLEAN"),
            ("e", "BOOL"),
            ("f", "LOGICAL"),
            ("g", "BLOB"),
            ("h", "BYTEA"),
            ("i", "BINARY"),
            ("j", "VARBINARY"),
            ("k", "DATE"),
            ("l", "DOUBLE"),
            ("m", "FLOAT8"),
            ("n", "NUMERIC"),
            ("o", "DECIMAL"),
            ("p", "DECIMAL(10, 3)"),
            ("q", "INTEGER"),
            ("r", "INT4"),
            ("s", "INT"),
            ("t", "SIGNED"),
            ("u", "INTERVAL"),
            ("v", "REAL"),
            ("w", "FLOAT4"),
            ("x", "FLOAT"),
            ("y", "SMALLINT"),
            ("z", "INT2"),
            ("A", "SHORT"),
            ("B", "TIME"),
            ("C", "TIMESTAMP"),
            ("D", "DATETIME"),
            ("E", "TINYINT"),
            ("F", "INT1"),
            ("G", "UBIGINT"),
            ("H", "UINTEGER"),
            ("I", "USMALLINT"),
            ("J", "UTINYINT"),
            ("K", "UUID"),
            ("L", "VARCHAR"),
            ("M", "CHAR"),
            ("N", "BPCHAR"),
            ("O", "TEXT"),
            ("P", "STRING"),
            ("Q", "LIST<INTEGER>"),
            ("R", "MAP<STRING, BIGINT>"),
            ("S", "STRUCT(a INT, b TEXT, c LIST<MAP<TEXT, LIST<FLOAT8>>>)"),
            ("T", "LIST<LIST<INTEGER>>"),
            ("U", "INTEGER[][]"),
            ("V", "TIMESTAMP_TZ"),
            ("W", "TIMESTAMP_SEC"),
            ("X", "TIMESTAMP_MS"),
            ("Y", "TIMESTAMP_NS"),
            ("Z", "JSON"),
        ]
    ],
)
def test_parser(column, type):
    ty = parse(type)
    assert ty == EXPECTED_SCHEMA[column]


def test_parse_type_warns():
    with pytest.warns(FutureWarning):
        parse_type("BIGINT")
