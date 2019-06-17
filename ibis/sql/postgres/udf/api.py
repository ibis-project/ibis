import inspect
from decimal import Decimal
import collections
import itertools

import sqlalchemy

import ibis.expr.rules as rlz
from ibis.expr import datatypes
from ibis.expr.signature import Argument as Arg
from ibis.sql.postgres.compiler import PostgresUDFNode, add_operation


_udf_name_cache = collections.defaultdict(itertools.count)


# type mapping based on: https://www.postgresql.org/docs/10/plpython-data.html
sql_default_type = 'VARCHAR'

pytype_sql = {
    bool: 'BOOLEAN',
    int: "INTEGER",
    float: 'DOUBLE',
    Decimal: 'NUMERIC',
    bytes: 'BYTEA',
    str: sql_default_type,
}

pytype_ibistype = {
    bool: datatypes.Boolean(),
    int: datatypes.Int32(),
    float: datatypes.Float(),
    Decimal: datatypes.Float(),
    bytes: datatypes.Binary(),
    str: datatypes.String(),
}

ibistype_pytype = {v: k for k, v in pytype_ibistype.items()}


def get_sqltype(type_):
    """Map input to the string specifying the Postgres data type
    (as is used in SQL defining UDF signatures)

    :param type_: str, a Python data type,
                  or an ibis DataType (instance or class)
    :return: string symbol of Postgres data type
    """
    if isinstance(type_, str) and type_ in pytype_sql.values():
        return type_
    elif type_ in pytype_sql.keys():
        return pytype_sql[type_]
    elif type_ in ibistype_pytype:
        return pytype_sql[ibistype_pytype[type_]]
    elif type_ in set(map(type, ibistype_pytype.keys())):
        return pytype_sql[ibistype_pytype[type_()]]
    else:
        raise ValueError(
            "Postgres data type not defined for: {}".format(type_)
        )


def create_udf_node(name, fields):
    """Create a new UDF node type.

    Parameters
    ----------
    name : str
        Then name of the UDF node
    fields : OrderedDict
        Mapping of class member name to definition

    Returns
    -------
    result : type
        A new PostgresUDFNode subclass
    """
    definition = next(_udf_name_cache[name])
    external_name = '{}_{:d}'.format(name, definition)
    return type(external_name, (PostgresUDFNode,), fields)


def existing_udf(name,
                 input_types,
                 output_type,
                 schema=None,
                 parameter_names=None):
    """Create a ibis function that refers to an existing Postgres UDF already
    defined in database

    Parameters
    ----------
    name: str
    input_types : List[DataType]
    output_type : DataType
    schema: str - optionally specify the schema that the UDF is defined in
    parameter_names: str - give names to the arguments of the UDF

    Returns
    -------
    wrapper : Callable
        The wrapped function
    """
    if parameter_names is None:
        parameter_names = ['v{}'.format(i) for i in range(len(input_types))]
    else:
        assert len(input_types) == len(parameter_names)

    udf_node_fields = collections.OrderedDict([
        (name, Arg(rlz.value(type_)))
        for name, type_ in zip(parameter_names, input_types)
    ] + [
        (
            'output_type',
            lambda self, output_type=output_type: rlz.shape_like(
                self.args, dtype=output_type
            )
        )
    ])
    udf_node_fields['resolve_name'] = lambda self: name

    udf_node = create_udf_node(name, udf_node_fields)

    def _translate_udf(t, expr):
        func_obj = sqlalchemy.func
        if schema is not None:
            func_obj = getattr(func_obj, schema)
        func_obj = getattr(func_obj, name)

        sa_args = [t.translate(arg) for arg in expr.op().args]

        return func_obj(*sa_args)

    add_operation(udf_node, _translate_udf)

    def wrapped(*args, **kwargs):
        node = udf_node(*args, **kwargs)
        return node.to_expr()

    return wrapped


def func_to_udf(conn,
                python_func,
                in_types=None,
                out_type=None,
                schema=None,
                name=None,
                overwrite=False):
    """Defines a UDF in the database

    Parameters
    ----------
    conn: sqlalchemy engine
    python_func: python function
    in_types: List[DataType]; if left None, will try to infer datatypes from
    function signature
    out_type : DataType
    schema: str - optionally specify the schema in which to define the UDF
    name: str - name for the UDF to be defined in database
    overwrite: bool - replace UDF in database if already exists

    Returns
    -------
    wrapper : Callable
        The ibis UDF object as a wrapped function
    """
    if name is None:
        internal_name = python_func.__name__
    else:
        internal_name = name
    signature = inspect.signature(python_func)
    parameter_names = signature.parameters.keys()
    if in_types is None:
        raise NotImplementedError('inferring in_types not implemented')
    if out_type is None:
        raise NotImplementedError('inferring out_type not implemented')
    replace = ' OR REPLACE ' if overwrite else ''
    schema_fragment = (schema + '.') if schema else ''
    template = """CREATE {replace} FUNCTION
{schema_fragment}{name}({signature})
RETURNS {return_type}
LANGUAGE plpythonu
AS $$
{func_definition}
return {internal_name}({args})
$$;
"""

    postgres_signature = ', '.join(
        '{name} {type}'.format(
            name=name,
            type=get_sqltype(type_),
        )
        for name, type_ in zip(parameter_names, in_types)
    )
    return_type = get_sqltype(out_type)
    func_definition = inspect.getsource(python_func)
    formatted_sql = template.format(
        replace=replace,
        schema_fragment=schema_fragment,
        name=internal_name,
        signature=postgres_signature,
        return_type=return_type,
        func_definition=func_definition,
        # for internal_name, need to make sure this works if passing
        # name parameter
        internal_name=python_func.__name__,
        args=', '.join(parameter_names)
    )
    conn.execute(formatted_sql)
    return existing_udf(
        name=internal_name,
        input_types=in_types,
        output_type=out_type,
        schema=schema,
        parameter_names=parameter_names
    )
