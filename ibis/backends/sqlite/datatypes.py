from ibis.expr.datatypes import (
    DataType,
    null,
    string,
    int64,
    float64,
    binary,
    decimal
)


def parse(text: str) -> DataType:
    text = text.strip().upper()

    if text == "":
        return null

    # SQLite affinity rules
    # (see https://www.sqlite.org/datatype3.html).

    # 1. If the declared type contains the string "INT" then it is
    # assigned INTEGER affinity.
    if "INT" in text:
        return int64

    # 2. If the declared type of the column contains any of the
    # strings "CHAR", "CLOB", or "TEXT" then that column has TEXT
    # affinity. Notice that the type VARCHAR contains the string
    # "CHAR" and is thus assigned TEXT affinity.
    if "CHAR" in text or "CLOB" in text or "TEXT" in text:
        return string

    # 3. If the declared type for a column contains the string "BLOB"
    # or if no type is specified then the column has affinity BLOB.
    if "BLOB" in text:
        return binary

    # 4. If the declared type for a column contains any of the strings
    # "REAL", "FLOA", or "DOUB" then the column has REAL affinity.
    if "REAL" in text or "FLOA" in text or "DOUB" in text:
        return float64

    # 5. Otherwise, the affinity is NUMERIC.
    return decimal
