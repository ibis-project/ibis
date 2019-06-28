"""Test support for already-defined UDFs in Postgres"""

import pytest

import sqlalchemy.exc
import ibis.expr.datatypes
import ibis.sql.postgres.udf.api


datatypes = ibis.expr.datatypes

# Database setup (tables and UDFs)


test_schema = 'test'

table_name = 'udf_test_users'

sql_table_setup = """DROP TABLE IF EXISTS {schema}.{table_name};
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
""".format(table_name=table_name, schema=test_schema)

sql_create_plpython = "CREATE EXTENSION plpythonu "

sql_define_py_udf = """CREATE OR REPLACE FUNCTION {schema}.pylen(x varchar)
RETURNS integer
LANGUAGE plpythonu
AS
$$
return len(x)
$$;""".format(schema=test_schema)

sql_define_udf = """CREATE OR REPLACE FUNCTION {schema}.custom_len(x varchar)
RETURNS integer
LANGUAGE SQL
AS
$$
SELECT length(x);
$$;""".format(schema=test_schema)


@pytest.fixture
def con_for_udf(con):
    con.con.execute("DROP SCHEMA IF EXISTS {} CASCADE".format(test_schema))
    con.con.execute("CREATE SCHEMA {}".format(test_schema))
    con.con.execute(sql_table_setup)
    con.con.execute(sql_define_udf)
    try:
        con.con.execute(sql_create_plpython)
    except sqlalchemy.exc.ProgrammingError as e:
        if '"plpythonu" already exists' not in str(e):
            raise Exception('PL/Python extension creation failed') from e
    con.con.execute(sql_define_py_udf)
    yield con
    # teardown
    con.con.execute("DROP SCHEMA IF EXISTS {} CASCADE".format(test_schema))


@pytest.fixture
def table(con_for_udf):
    return con_for_udf.table(table_name, schema=test_schema)


# Create ibis UDF objects referring to UDFs already created in the database

custom_length_udf = ibis.sql.postgres.udf.api.existing_udf(
    'custom_len',
    input_types=[ibis.expr.datatypes.String()],
    output_type=ibis.expr.datatypes.Integer(),
    schema=test_schema
)

py_length_udf = ibis.sql.postgres.udf.api.existing_udf(
    'pylen',
    input_types=[ibis.expr.datatypes.String()],
    output_type=ibis.expr.datatypes.Integer(),
    schema=test_schema
)


# Tests

def test_sql_length_udf_worked(table):
    """Test creating ibis UDF object based on existing UDF in the database"""
    result_obj = table[
        table,
        custom_length_udf(table['user_name']).name('custom_len')
    ]
    result = result_obj.execute()
    assert result['custom_len'].sum() == result['name_length'].sum()


def test_py_length_udf_worked(table):
    result_obj = table[
        table,
        py_length_udf(table['user_name']).name('custom_len')
    ]
    result = result_obj.execute()
    assert result['custom_len'].sum() == result['name_length'].sum()


def mult_a_b(a, b):
    """Test function to be defined in-database as a UDF
    and used via ibis UDF"""
    return a * b


def test_func_to_udf_smoke(con_for_udf, table):
    """Test creating a UDF in database based on Python function
    and then creating an ibis UDF object based on that"""
    mult_a_b_udf = ibis.sql.postgres.udf.api.func_to_udf(
        con_for_udf.con,
        mult_a_b,
        (datatypes.Int32(), datatypes.Int32()),
        datatypes.Int32(),
        schema=test_schema,
        overwrite=True
    )
    table_filt = table.filter(table['user_id'] == 2)
    expr = table_filt[
        mult_a_b_udf(
            table_filt['user_id'],
            table_filt['name_length']
        ).name('mult_result')
    ]
    result = expr.execute()
    assert result['mult_result'].iloc[0] == 8
