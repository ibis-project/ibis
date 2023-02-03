"""Test support for already-defined UDFs in Postgres."""

import functools

import pytest

import ibis
import ibis.expr.datatypes as dt
from ibis.util import guid

pytest.importorskip("psycopg2")
sa = pytest.importorskip("sqlalchemy")

from ibis.backends.postgres.udf import PostgresUDFError, existing_udf, udf  # noqa: E402


@pytest.fixture(scope='session')
def test_schema(con):
    schema_name = f'udf_test_{guid()}'
    with con.begin() as c:
        c.exec_driver_sql(f"CREATE SCHEMA IF NOT EXISTS {schema_name}")
    yield schema_name
    with con.begin() as c:
        c.exec_driver_sql(f"DROP SCHEMA IF EXISTS {schema_name} CASCADE")


@pytest.fixture(scope='session')
def table_name():
    return 'udf_test_users'


@pytest.fixture(scope='session')
def sql_table_setup(test_schema, table_name):
    return f"""\
DROP TABLE IF EXISTS {test_schema}.{table_name};
CREATE TABLE {test_schema}.{table_name} (
    user_id integer,
    user_name varchar,
    name_length integer
);
INSERT INTO {test_schema}.{table_name} VALUES
(1, 'Raj', 3),
(2, 'Judy', 4),
(3, 'Jonathan', 8)"""


@pytest.fixture(scope='session')
def sql_define_py_udf(test_schema):
    return f"""\
CREATE OR REPLACE FUNCTION {test_schema}.pylen(x varchar)
RETURNS integer
LANGUAGE plpython3u
AS
$$
return len(x)
$$"""


@pytest.fixture(scope='session')
def sql_define_udf(test_schema):
    return f"""\
CREATE OR REPLACE FUNCTION {test_schema}.custom_len(x varchar)
RETURNS integer
LANGUAGE SQL
AS
$$
SELECT length(x);
$$"""


@pytest.fixture(scope='session')
@pytest.mark.usefixtures("test_schema")
def con_for_udf(con, sql_table_setup, sql_define_udf, sql_define_py_udf):
    with con.begin() as c:
        c.exec_driver_sql(sql_table_setup)
        c.exec_driver_sql(sql_define_udf)
        c.exec_driver_sql(sql_define_py_udf)
    yield con


@pytest.fixture
def table(con_for_udf, table_name, test_schema):
    return con_for_udf.table(table_name, schema=test_schema)


# Tests


def test_existing_sql_udf(test_schema, table):
    """Test creating ibis UDF object based on existing UDF in the database."""
    # Create ibis UDF objects referring to UDFs already created in the database
    custom_length_udf = existing_udf(
        'custom_len',
        input_types=[dt.string],
        output_type=dt.int32,
        schema=test_schema,
    )
    result_obj = table[table, custom_length_udf(table['user_name']).name('custom_len')]
    result = result_obj.execute()
    assert result['custom_len'].sum() == result['name_length'].sum()


def test_existing_plpython_udf(test_schema, table):
    # Create ibis UDF objects referring to UDFs already created in the database
    py_length_udf = existing_udf(
        'pylen',
        input_types=[dt.string],
        output_type=dt.int32,
        schema=test_schema,
    )
    result_obj = table[table, py_length_udf(table['user_name']).name('custom_len')]
    result = result_obj.execute()
    assert result['custom_len'].sum() == result['name_length'].sum()


def mult_a_b(a, b):
    """Test function to be defined in-database as a UDF and used via ibis
    UDF."""
    return a * b


def test_udf(con_for_udf, test_schema, table):
    """Test creating a UDF in database based on Python function and then
    creating an ibis UDF object based on that."""
    mult_a_b_udf = udf(
        con_for_udf,
        mult_a_b,
        (dt.int32, dt.int32),
        dt.int32,
        schema=test_schema,
        replace=True,
        language="plpython3u",
    )
    table_filt = table.filter(table['user_id'] == 2)
    expr = table_filt[
        mult_a_b_udf(table_filt['user_id'], table_filt['name_length']).name(
            'mult_result'
        )
    ]
    result = expr.execute()
    assert result['mult_result'].iloc[0] == 8


def pysplit(text, split):
    return text.split(split)


def test_array_type(con_for_udf, test_schema, table):
    """Test that usage of Array types work Other scalar types can be
    represented either by the class or an instance, but Array types work
    differently.

    Array types must be an instance, because the Array class must be
    instantiated specifying the datatype of the elements of the array.
    """
    pysplit_udf = udf(
        con_for_udf,
        pysplit,
        (dt.string, dt.string),
        dt.Array(dt.string),
        schema=test_schema,
        replace=True,
        language="plpython3u",
    )
    splitter = ibis.literal(' ', dt.string)
    result = pysplit_udf(table['user_name'], splitter).name('split_name')
    result.execute()


def test_client_udf_api(con_for_udf, test_schema, table):
    """Test creating a UDF in database based on Python function using an ibis
    client method."""

    def multiply(a, b):
        return a * b

    multiply_udf = con_for_udf.udf(
        multiply,
        [dt.int32, dt.int32],
        dt.int32,
        schema=test_schema,
        replace=True,
        language="plpython3u",
    )

    table_filt = table.filter(table['user_id'] == 2)
    expr = table_filt[
        multiply_udf(table_filt['user_id'], table_filt['name_length']).name(
            'mult_result'
        )
    ]
    result = expr.execute()
    assert result['mult_result'].iloc[0] == 8


def test_client_udf_decorator_fails(con_for_udf, test_schema):
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
    def multiply(a, b):
        return a * b

    with pytest.raises(PostgresUDFError):
        con_for_udf.udf(
            multiply,
            [dt.int32, dt.int32],
            dt.int32,
            schema=test_schema,
            replace=True,
            language="plpython3u",
        )
