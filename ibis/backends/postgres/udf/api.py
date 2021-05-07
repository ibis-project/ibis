import collections
import inspect
import itertools
from textwrap import dedent

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import dialect as sa_postgres_dialect

import ibis.expr.rules as rlz
import ibis.udf.validate as v
from ibis import IbisError
from ibis.backends.base_sqlalchemy.alchemy import _to_sqla_type
from ibis.backends.postgres.compiler import (
    PostgreSQLExprTranslator,
    PostgresUDFNode,
    add_operation,
)
from ibis.expr.signature import Argument as Arg

_udf_name_cache = collections.defaultdict(itertools.count)


class PostgresUDFError(IbisError):
    pass


def ibis_to_pg_sa_type(ibis_type):
    """Map an ibis DataType to a Postgres-compatible sqlalchemy type"""
    return _to_sqla_type(
        ibis_type, type_map=PostgreSQLExprTranslator._type_map
    )


def sa_type_to_postgres_str(sa_type):
    """Map a Postgres-compatible sqlalchemy type to a Postgres-appropriate
    string"""
    if callable(sa_type):
        sa_type = sa_type()
    return sa_type.compile(dialect=sa_postgres_dialect())


def ibis_to_postgres_str(ibis_type):
    """Map an ibis DataType to a Postgres-appropriate string"""
    return sa_type_to_postgres_str(ibis_to_pg_sa_type(ibis_type))


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
    type
        A new PostgresUDFNode subclass
    """
    definition = next(_udf_name_cache[name])
    external_name = '{}_{:d}'.format(name, definition)
    return type(external_name, (PostgresUDFNode,), fields)


def existing_udf(name, input_types, output_type, schema=None, parameters=None):
    """Create an ibis function that refers to an existing Postgres UDF already
    defined in database

    Parameters
    ----------
    name: str
    input_types : List[DataType]
    output_type : DataType
    schema: str - optionally specify the schema that the UDF is defined in
    parameters: List[str] - give names to the arguments of the UDF

    Returns
    -------
    Callable
        The wrapped function
    """

    if parameters is None:
        parameters = ['v{}'.format(i) for i in range(len(input_types))]
    elif len(input_types) != len(parameters):
        raise ValueError(
            (
                "Length mismatch in arguments to existing_udf: "
                "len(input_types)={}, len(parameters)={}"
            ).format(len(input_types), len(parameters))
        )

    v.validate_output_type(output_type)

    udf_node_fields = collections.OrderedDict(
        [
            (name, Arg(rlz.value(type_)))
            for name, type_ in zip(parameters, input_types)
        ]
        + [
            (
                'output_type',
                lambda self, output_type=output_type: rlz.shape_like(
                    self.args, dtype=output_type
                ),
            )
        ]
    )
    udf_node_fields['resolve_name'] = lambda self: name

    udf_node = create_udf_node(name, udf_node_fields)

    def _translate_udf(t, expr):
        func_obj = sa.func
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


def udf(
    client,
    python_func,
    in_types,
    out_type,
    schema=None,
    replace=False,
    name=None,
):
    """Defines a UDF in the database

    Parameters
    ----------
    client: PostgreSQLClient
    python_func: python function
    in_types: List[DataType]
    out_type : DataType
    schema: str - optionally specify the schema in which to define the UDF
    replace: bool - replace UDF in database if already exists
    name: str - name for the UDF to be defined in database

    Returns
    -------
    Callable

        The ibis UDF object as a wrapped function
    """
    if name is None:
        internal_name = python_func.__name__
    else:
        internal_name = name
    signature = inspect.signature(python_func)
    parameter_names = signature.parameters.keys()
    replace_text = ' OR REPLACE ' if replace else ''
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
        '{name} {type}'.format(name=name, type=ibis_to_postgres_str(type_))
        for name, type_ in zip(parameter_names, in_types)
    )
    return_type = ibis_to_postgres_str(out_type)
    # If function definition is indented extra,
    # Postgres UDF will fail with indentation error.
    func_definition = dedent(inspect.getsource(python_func))
    if func_definition.strip().startswith('@'):
        raise PostgresUDFError(
            'Use of decorators on a function to be turned into Postgres UDF '
            'is not supported. The body of the UDF must be wholly '
            'self-contained. '
        )
        # Since the decorator syntax does not first bind
        # the function name to the wrapped function but instead includes
        # the decorator(s). Therefore, the decorators themselves will
        # be included in the string coming from `inspect.getsource()`.
        # Since the decorator objects are not defined, execution of the
        # UDF results in a NameError.

    formatted_sql = template.format(
        replace=replace_text,
        schema_fragment=schema_fragment,
        name=internal_name,
        signature=postgres_signature,
        return_type=return_type,
        func_definition=func_definition,
        # for internal_name, need to make sure this works if passing
        # name parameter
        internal_name=python_func.__name__,
        args=', '.join(parameter_names),
    )
    client.con.execute(formatted_sql)
    return existing_udf(
        name=internal_name,
        input_types=in_types,
        output_type=out_type,
        schema=schema,
        parameters=parameter_names,
    )
