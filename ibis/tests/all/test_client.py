import pytest
from pkg_resources import parse_version

import ibis
import ibis.expr.datatypes as dt
from ibis.tests.backends import BigQuery, Impala, PySpark, Spark


@pytest.fixture
def new_schema():
    return ibis.schema([('a', 'string'), ('b', 'bool'), ('c', 'int32')])


@pytest.mark.xfail_unsupported
def test_version(backend, con):
    expected_type = (
        type(parse_version('1.0')),
        type(parse_version('1.0-legacy')),
    )
    assert isinstance(con.version, expected_type)


@pytest.mark.parametrize(
    ('expr_fn', 'expected'),
    [
        (lambda t: t.string_col, [('string_col', dt.String)]),
        (
            lambda t: t[t.string_col, t.bigint_col],
            [('string_col', dt.String), ('bigint_col', dt.Int64)],
        ),
    ],
)
def test_query_schema(backend, con, alltypes, expr_fn, expected):
    if not hasattr(con, '_build_ast'):
        pytest.skip(
            '{} backend has no _build_ast method'.format(
                type(backend).__name__
            )
        )

    expr = expr_fn(alltypes)

    # we might need a public API for it
    ast = con._build_ast(expr, backend.make_context())
    query = con.query_class(con, ast)
    schema = query.schema()

    # clickhouse columns has been defined as non-nullable
    # whereas other backends don't support non-nullable columns yet
    expected = ibis.schema(
        [
            (name, dtype(nullable=schema[name].nullable))
            for name, dtype in expected
        ]
    )
    assert query.schema().equals(expected)


@pytest.mark.parametrize(
    'sql',
    [
        'select * from functional_alltypes limit 10',
        'select * from functional_alltypes \nlimit 10\n',
    ],
)
@pytest.mark.xfail_backends((BigQuery,))
@pytest.mark.xfail_unsupported
def test_sql(backend, con, sql):
    if not hasattr(con, 'sql') or not hasattr(con, '_get_schema_using_query'):
        pytest.skip('Backend {} does not support sql method'.format(backend))

    # execute the expression using SQL query
    con.sql(sql).execute()


# test table


@pytest.mark.xfail_unsupported
def test_create_table_from_schema(con, backend, new_schema, temp_table):
    if not hasattr(con, 'create_table') or not hasattr(con, 'drop_table'):
        pytest.xfail(
            '{} backend doesn\'t have create_table or drop_table methods.'
        )

    con.create_table(temp_table, schema=new_schema)

    t = con.table(temp_table)

    for k, i_type in t.schema().items():
        assert new_schema[k] == i_type


@pytest.mark.xfail_unsupported
def test_rename_table(con, backend, temp_table, new_schema):
    if not hasattr(con, 'rename_table'):
        pytest.xfail('{} backend doesn\'t have rename_table method.')

    temp_table_original = '{}_original'.format(temp_table)
    con.create_table(temp_table_original, schema=new_schema)

    t = con.table(temp_table_original)
    t.rename(temp_table)

    assert con.table(temp_table) is not None
    assert temp_table in con.list_tables()


@pytest.mark.xfail_unsupported
@pytest.mark.xfail_backends([Impala, PySpark, Spark])
def test_nullable_input_output(con, backend, temp_table):
    # - Impala, PySpark and Spark non-nullable issues #2138 and #2137
    if not hasattr(con, 'create_table') or not hasattr(con, 'drop_table'):
        pytest.xfail(
            '{} backend doesn\'t have create_table or drop_table methods.'
        )

    sch = ibis.schema(
        [
            ('foo', 'int64'),
            ('bar', ibis.expr.datatypes.int64(nullable=False)),
            ('baz', 'boolean*'),
        ]
    )

    con.create_table(temp_table, schema=sch)

    t = con.table(temp_table)

    assert t.schema().types[0].nullable
    assert not t.schema().types[1].nullable
    assert t.schema().types[2].nullable


# view tests


@pytest.mark.xfail_unsupported
@pytest.mark.xfail_backends([PySpark, Spark])
def test_create_drop_view(con, backend, temp_view):
    # pyspark and spark skipt because table actually is a temporary view
    if not hasattr(con, 'create_view') or not hasattr(con, 'drop_view'):
        pytest.xfail(
            '{} backend doesn\'t have create_view or drop_view methods.'
        )

    # setup
    table_name = 'functional_alltypes'
    expr = con.table(table_name).limit(1)

    # create a new view
    con.create_view(temp_view, expr)
    # check if the view was created
    assert temp_view in con.list_tables()

    t_expr = con.table(table_name)
    v_expr = con.table(temp_view)
    # check if the view and the table has the same fields
    assert set(t_expr.schema().names) == set(v_expr.schema().names)
