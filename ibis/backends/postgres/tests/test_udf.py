"""Test support for already-defined UDFs in Postgres"""

import functools

import pytest

import ibis
import ibis.expr.datatypes as dt
from ibis.sql.postgres import existing_udf, udf
from ibis.sql.postgres.udf.api import PostgresUDFError

# mark test module as postgresql (for ability to easily exclude,
# e.g. in conda build tests)
# (Temporarily adding `postgis` marker so Azure Windows pipeline will exclude
#     pl/python tests.
#     TODO: update Windows pipeline to exclude postgres_extensions
#     TODO: remove postgis marker below once Windows pipeline updated
pytestmark = [
    pytest.mark.postgres,
    pytest.mark.udf,
    pytest.mark.postgis,
    pytest.mark.postgres_extensions,
]

# Database setup (tables and UDFs)


@pytest.fixture(scope='session')
def next_serial(con):
    # `test_sequence` SEQUENCE is created in database in the
    # load-data.sh --> datamgr.py#postgres step
    # to avoid parallel attempts to create the same sequence (when testing
    # run in parallel
    serial_proxy = con.con.execute("SELECT nextval('test_sequence') as value;")
    return serial_proxy.fetchone()['value']


@pytest.fixture(scope='session')
def test_schema(con, next_serial):
    schema_name = 'udf_test_{}'.format(next_serial)
    con.con.execute("CREATE SCHEMA IF NOT EXISTS {};".format(schema_name))
    return schema_name


@pytest.fixture(scope='session')
def table_name():
    return 'udf_test_users'


@pytest.fixture(scope='session')
def sql_table_setup(test_schema, table_name):
    return """DROP TABLE IF EXISTS {schema}.{table_name};
CREATE TABLE {schema}.{table_name} (
    user_id integer,
    user_name varchar,
    name_length integer
);
INSERT INTO {schema}.{table_name} VALUES
(1, 'Raj', 3),
(2, 'Judy', 4),
(3, 'Jonathan', 8)
;
""".format(
        schema=test_schema, table_name=table_name
    )


@pytest.fixture(scope='session')
def sql_define_py_udf(test_schema):
    return """CREATE OR REPLACE FUNCTION {schema}.pylen(x varchar)
RETURNS integer
LANGUAGE plpythonu
AS
$$
return len(x)
$$;""".format(
        schema=test_schema
    )


@pytest.fixture(scope='session')
def sql_define_udf(test_schema):
    return """CREATE OR REPLACE FUNCTION {schema}.custom_len(x varchar)
RETURNS integer
LANGUAGE SQL
AS
$$
SELECT length(x);
$$;""".format(
        schema=test_schema
    )


@pytest.fixture(scope='session')
def con_for_udf(
    con, test_schema, sql_table_setup, sql_define_udf, sql_define_py_udf
):
    con.con.execute(sql_table_setup)
    con.con.execute(sql_define_udf)
    con.con.execute(sql_define_py_udf)
    try:
        yield con
    finally:
        # teardown
        con.con.execute("DROP SCHEMA IF EXISTS {} CASCADE".format(test_schema))


@pytest.fixture
def table(con_for_udf, table_name, test_schema):
    return con_for_udf.table(table_name, schema=test_schema)


# Tests


def test_existing_sql_udf(test_schema, table):
    """Test creating ibis UDF object based on existing UDF in the database"""
    # Create ibis UDF objects referring to UDFs already created in the database
    custom_length_udf = existing_udf(
        'custom_len',
        input_types=[dt.string],
        output_type=dt.int32,
        schema=test_schema,
    )
    result_obj = table[
        table, custom_length_udf(table['user_name']).name('custom_len')
    ]
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
    result_obj = table[
        table, py_length_udf(table['user_name']).name('custom_len')
    ]
    result = result_obj.execute()
    assert result['custom_len'].sum() == result['name_length'].sum()


def mult_a_b(a, b):
    """Test function to be defined in-database as a UDF
    and used via ibis UDF"""
    return a * b


def test_udf(con_for_udf, test_schema, table):
    """Test creating a UDF in database based on Python function
    and then creating an ibis UDF object based on that"""
    mult_a_b_udf = udf(
        con_for_udf,
        mult_a_b,
        (dt.int32, dt.int32),
        dt.int32,
        schema=test_schema,
        replace=True,
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
    """Test that usage of Array types work
    Other scalar types can be represented either by the class or an instance,
    but Array types work differently. Array types must be an instance,
    because the Array class must be instantiated specifying the datatype
    of the elements of the array.
    """
    pysplit_udf = udf(
        con_for_udf,
        pysplit,
        (dt.string, dt.string),
        dt.Array(dt.string),
        schema=test_schema,
        replace=True,
    )
    splitter = ibis.literal(' ', dt.string)
    result = pysplit_udf(table['user_name'], splitter).name('split_name')
    result.execute()


def test_client_udf_api(con_for_udf, test_schema, table):
    """Test creating a UDF in database based on Python function
    using an ibis client method."""

    def multiply(a, b):
        return a * b

    multiply_udf = con_for_udf.udf(
        multiply,
        [dt.int32, dt.int32],
        dt.int32,
        schema=test_schema,
        replace=True,
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
    function that has been defined with decorators. Decorators are not
    currently supported, because the decorators end up in the body of the UDF
    but are not defined in the body, therefore causing a NameError."""

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
        )
