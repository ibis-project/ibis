from __future__ import annotations

import pytest
from pytest import param

from ibis import udf


@udf.scalar.builtin
def soundex(s: str) -> str:
    ...


@udf.scalar.builtin
def jarowinkler_similarity(a: str, b: str) -> float:
    ...


# TODO: allow multiple signatures
@udf.scalar.builtin(name="compress")
def compress_str(data: str, method: str) -> bytes:
    ...


@udf.scalar.builtin(name="compress")
def compress_bytes(data: bytes, method: str) -> bytes:
    ...


@pytest.mark.parametrize(
    ("func", "args"),
    [
        param(soundex, ("snow",), id="soundex"),
        param(jarowinkler_similarity, ("snow", "show"), id="jarowinkler_similarity"),
    ],
)
def test_builtin(con, func, args):
    expr = func(*args)

    query = f"SELECT {func.__name__}({', '.join(map(repr, args))})"
    with con.begin() as c:
        expected = c.exec_driver_sql(query).scalar()

    assert con.execute(expr) == expected


@pytest.mark.parametrize(
    ("func", "pyargs", "snowargs"),
    [
        param(compress_str, ("snow", "zstd"), ("'snow'", "'zstd'"), id="str"),
        param(compress_bytes, (b"snow", "zstd"), ("'snow'", "'zstd'"), id="bytes"),
    ],
)
def test_compress(con, func, pyargs, snowargs):
    expr = func(*pyargs)

    query = f"SELECT compress({', '.join(snowargs)})"
    with con.begin() as c:
        expected = c.exec_driver_sql(query).scalar()

    assert con.execute(expr) == expected
