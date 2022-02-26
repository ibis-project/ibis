import pytest

import ibis
from ibis.backends.duckdb.parser import parse_type

PAIRS = """\
a BIGINT,
b INT8,
c LONG,
d BOOLEAN,
e BOOL,
f LOGICAL,
g BLOB,
h BYTEA,
i BINARY,
j VARBINARY,
k DATE,
l DOUBLE,
m FLOAT8,
n NUMERIC,
o DECIMAL,
p DECIMAL(10, 3),
q INTEGER,
r INT4,
s INT,
t SIGNED,
u INTERVAL,
v REAL,
w FLOAT4,
x FLOAT,
y SMALLINT,
z INT2,
aa SHORT,
ab TIME,
ac TIMESTAMP,
ad DATETIME,
ae TINYINT,
af INT1,
ag UBIGINT,
ah UINTEGER,
ai USMALLINT,
aj UTINYINT,
ak UUID,
al VARCHAR,
am CHAR,
an BPCHAR,
ao TEXT,
ap STRING"""

EXPECTED_SCHEMA = ibis.schema(
    [
        ("a", "int64"),
        ("b", "int64"),
        ("c", "int64"),
        ("d", "boolean"),
        ("e", "boolean"),
        ("f", "boolean"),
        ("g", "binary"),
        ("h", "binary"),
        ("i", "binary"),
        ("j", "binary"),
        ("k", "date"),
        ("l", "float64"),
        ("m", "float64"),
        ("n", "decimal(18, 3)"),
        ("o", "decimal(18, 3)"),
        ("p", "decimal(10, 3)"),
        ("q", "int32"),
        ("r", "int32"),
        ("s", "int32"),
        ("t", "int32"),
        ("u", "interval"),
        ("v", "float32"),
        ("w", "float32"),
        ("x", "float32"),
        ("y", "int16"),
        ("z", "int16"),
        ("aa", "int16"),
        ("ab", "time"),
        ("ac", "timestamp('UTC')"),
        ("ad", "timestamp('UTC')"),
        ("ae", "int8"),
        ("af", "int8"),
        ("ag", "uint64"),
        ("ah", "uint32"),
        ("ai", "uint16"),
        ("aj", "uint8"),
        ("ak", "uuid"),
        ("al", "string"),
        ("am", "string"),
        ("an", "string"),
        ("ao", "string"),
        ("ap", "string"),
    ]
)


@pytest.mark.parametrize(
    ("column", "type"),
    [
        pytest.param(name, type, id=type.lower())
        for name, type in (
            tuple(pair.rstrip(",").split(" ", 1))
            for pair in PAIRS.splitlines()
        )
    ],
)
def test_parser(column, type):
    ty = parse_type(type)
    assert ty == EXPECTED_SCHEMA[column]
