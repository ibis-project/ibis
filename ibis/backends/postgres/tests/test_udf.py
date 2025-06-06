"""Test support for already-defined UDFs in Postgres."""

from __future__ import annotations

import functools
import sys

import pytest

import ibis
import ibis.common.exceptions as exc
import ibis.expr.datatypes as dt
from ibis import udf
from ibis.util import guid

pytest.importorskip("psycopg")


@pytest.fixture(scope="session")
def test_database(con):
    db_name = f"udf_test_{guid()}"
    con.create_database(db_name, force=True)
    yield db_name
    con.drop_database(db_name, force=True, cascade=True)


@pytest.fixture(scope="session")
def table_name():
    return "udf_test_users"


@pytest.fixture(scope="session")
def sql_table_setup(test_database, table_name):
    return f"""\
DROP TABLE IF EXISTS {test_database}.{table_name};
CREATE TABLE {test_database}.{table_name} (
    user_id integer,
    user_name varchar,
    name_length integer
);
INSERT INTO {test_database}.{table_name} VALUES
(1, 'Raj', 3),
(2, 'Judy', 4),
(3, 'Jonathan', 8)"""


@pytest.fixture(scope="session")
def sql_define_py_udf(test_database):
    return f"""\
CREATE OR REPLACE FUNCTION {test_database}.pylen(x varchar)
RETURNS integer
LANGUAGE plpython3u
AS
$$
return len(x)
$$"""


@pytest.fixture(scope="session")
def sql_define_udf(test_database):
    return f"""\
CREATE OR REPLACE FUNCTION {test_database}.custom_len(x varchar)
RETURNS integer
LANGUAGE SQL
AS
$$
SELECT length(x);
$$"""


@pytest.fixture(scope="session")
def con_for_udf(con, sql_table_setup, sql_define_udf, sql_define_py_udf, test_database):  # noqa: ARG001
    with con.begin() as c:
        c.execute(sql_table_setup)
        c.execute(sql_define_udf)
        c.execute(sql_define_py_udf)
    yield con


@pytest.fixture
def table(con_for_udf, table_name, test_database):
    return con_for_udf.table(table_name, database=test_database)


def test_existing_sql_udf(con_for_udf, test_database, table):
    """Test creating ibis UDF object based on existing UDF in the database."""
    # Create ibis UDF objects referring to UDFs already created in the database
    custom_length_udf = con_for_udf.function("custom_len", database=test_database)
    result_obj = table.select(
        table, custom_length_udf(table["user_name"]).name("custom_len")
    )
    result = result_obj.execute()
    assert result["custom_len"].sum() == result["name_length"].sum()


def test_existing_plpython_udf(con_for_udf, test_database, table):
    # Create ibis UDF objects referring to UDFs already created in the database
    py_length_udf = con_for_udf.function("pylen", database=test_database)
    result_obj = table.select(
        table, py_length_udf(table["user_name"]).name("custom_len")
    )
    result = result_obj.execute()
    assert result["custom_len"].sum() == result["name_length"].sum()


def test_udf(test_database, table):
    """Test creating a UDF in database based on Python function and then
    creating an ibis UDF object based on that."""

    @udf.scalar.python(database=test_database)
    def mult_a_b(a: int, b: int) -> int:
        return a * b

    table_filt = table.filter(table["user_id"] == 2)
    expr = table_filt.select(
        mult_result=mult_a_b(table_filt["user_id"], table_filt["name_length"])
    )
    result = expr.execute()
    assert result["mult_result"].iat[0] == 8


@pytest.mark.xfail(
    condition=sys.version_info[:2] < (3, 9),
    raises=TypeError,
    reason="no straightforward way to use new (Python 3.9) annotations syntax",
)
def test_array_type(test_database, table):
    """Test that usage of Array types work Other scalar types can be
    represented either by the class or an instance, but Array types work
    differently.

    Array types must be an instance, because the Array class must be
    instantiated specifying the datatype of the elements of the array.
    """

    @udf.scalar.python(database=test_database)
    def pysplit(text: str, split: str) -> list[str]:
        return text.split(split)

    splitter = ibis.literal(" ", dt.string)
    result = pysplit(table["user_name"], splitter)
    result.execute()


def test_client_udf_api(test_database, table):
    """Test creating a UDF in database based on Python function using an ibis
    client method."""

    @udf.scalar.python(database=test_database)
    def multiply(a: int, b: int) -> int:
        return a * b

    table_filt = table.filter(table["user_id"] == 2)
    expr = table_filt.select(
        mult_result=multiply(table_filt["user_id"], table_filt["name_length"])
    )
    result = expr.execute()
    assert result["mult_result"].iat[0] == 8


def test_client_udf_decorator_fails(con_for_udf, test_database):
    """Test that UDF creation fails when creating a UDF based on a Python
    function that has been defined with decorators.

    Decorators are not currently supported, because the decorators end
    up in the body of the UDF but are not defined in the body, therefore
    causing a NameError.
    """

    def decorator(f):
        @functools.wraps(f)
        def wrapped(*args, **kwds):
            return f(*args, **kwds)

        return wrapped

    @decorator
    @udf.scalar.python(database=test_database)
    def multiply(a: int, b: int) -> int:
        return a * b

    with pytest.raises(exc.InvalidDecoratorError, match="@udf"):
        con_for_udf.execute(multiply(1, 2))
