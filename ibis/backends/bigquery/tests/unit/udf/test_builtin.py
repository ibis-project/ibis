from __future__ import annotations

import ibis

to_sql = ibis.bigquery.compile


@ibis.udf.scalar.builtin
def farm_fingerprint(value: bytes) -> int: ...


@ibis.udf.scalar.builtin(schema="fn", database="bqutil")
def from_hex(value: str) -> int:
    """Community function to convert from hex string to integer.

    See:
    https://github.com/GoogleCloudPlatform/bigquery-utils/tree/master/udfs/community#from_hexvalue-string
    """


def test_bqutil_fn_from_hex(snapshot):
    # Project ID should be enclosed in backticks.
    expr = from_hex("face")
    snapshot.assert_match(to_sql(expr), "out.sql")


def test_farm_fingerprint(snapshot):
    # No backticks needed if there is no schema defined.
    expr = farm_fingerprint(b"Hello, World!")
    snapshot.assert_match(to_sql(expr), "out.sql")
