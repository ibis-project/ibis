from __future__ import annotations

import collections
import inspect
import itertools
from textwrap import dedent
from typing import TYPE_CHECKING, Any, Callable

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import dialect

import ibis
import ibis.expr.datatypes as dt
import ibis.expr.rules as rlz
from ibis import IbisError
from ibis.backends.postgres.compiler import PostgreSQLExprTranslator, PostgresUDFNode
from ibis.backends.postgres.datatypes import PostgresType
from ibis.legacy.udf.validate import validate_output_type

if TYPE_CHECKING:
    from collections.abc import MutableMapping, Sequence

_udf_name_cache: MutableMapping[str, Any] = collections.defaultdict(itertools.count)

_postgres_dialect = dialect()


class PostgresUDFError(IbisError):
    pass


def _ibis_to_postgres_str(ibis_type):
    """Map an ibis DataType to a Postgres-appropriate string."""
    satype = PostgresType.from_ibis(ibis_type)
    if callable(satype):
        satype = satype()
    return satype.compile(dialect=_postgres_dialect)


def _create_udf_node(
    name: str,
    fields: dict[str, Any],
) -> type[PostgresUDFNode]:
    """Create a new UDF node type.

    Parameters
    ----------
    name
        Then name of the UDF node
    fields
        Mapping of class member name to definition

    Returns
    -------
    type[PostgresUDFNode]
        A new PostgresUDFNode subclass
    """
    definition = next(_udf_name_cache[name])
    external_name = f"{name}_{definition:d}"
    return type(external_name, (PostgresUDFNode,), fields)


def existing_udf(name, input_types, output_type, schema=None, parameters=None):
    """Create an ibis function that refers to an existing Postgres UDF."""

    if parameters is None:
        parameters = [f"v{i}" for i in range(len(input_types))]
    elif len(input_types) != len(parameters):
        raise ValueError(
            (
                "Length mismatch in arguments to existing_udf: "
                "len(input_types)={}, len(parameters)={}"
            ).format(len(input_types), len(parameters))
        )

    validate_output_type(output_type)

    udf_node_fields = {
        name: rlz.ValueOf(type_) for name, type_ in zip(parameters, input_types)
    }
    udf_node_fields["name"] = name
    udf_node_fields["dtype"] = output_type

    udf_node = _create_udf_node(name, udf_node_fields)

    def _translate_udf(t, op):
        func_obj = sa.func
        if schema is not None:
            func_obj = getattr(func_obj, schema)
        func_obj = getattr(func_obj, name)

        sa_args = [t.translate(arg) for arg in op.args]

        return func_obj(*sa_args)

    PostgreSQLExprTranslator.add_operation(udf_node, _translate_udf)

    def wrapped(*args, **kwargs):
        node = udf_node(*args, **kwargs)
        return node.to_expr()

    return wrapped


def udf(
    client: ibis.backends.postgres.Backend,
    python_func: Callable[..., Any],
    in_types: Sequence[dt.DataType],
    out_type: dt.DataType,
    schema: str | None = None,
    replace: bool = False,
    name: str | None = None,
    language: str = "plpythonu",
):
    """Define a UDF in the database.

    Parameters
    ----------
    client
        A postgres Backend instance
    python_func
        Python function
    in_types
        Input DataTypes
    out_type
        Output DataType
    schema
        The postgres schema in which to define the UDF
    replace
        Replace UDF in database if already exists
    name
        Name for the UDF to be defined in database
    language
        The language to use for the UDF

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
    replace_text = " OR REPLACE " if replace else ""
    schema_fragment = (schema + ".") if schema else ""
    template = """CREATE {replace} FUNCTION
{schema_fragment}{name}({signature})
RETURNS {return_type}
LANGUAGE {language}
AS $$
{func_definition}
return {internal_name}({args})
$$;
"""

    postgres_signature = ", ".join(
        f"{name} {_ibis_to_postgres_str(type_)}"
        for name, type_ in zip(parameter_names, in_types)
    )
    return_type = _ibis_to_postgres_str(out_type)
    # If function definition is indented extra,
    # Postgres UDF will fail with indentation error.
    func_definition = dedent(inspect.getsource(python_func))
    if func_definition.strip().startswith("@"):
        raise PostgresUDFError(
            "Use of decorators on a function to be turned into Postgres UDF "
            "is not supported. The body of the UDF must be wholly "
            "self-contained. "
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
        language=language,
        func_definition=func_definition,
        # for internal_name, need to make sure this works if passing
        # name parameter
        internal_name=python_func.__name__,
        args=", ".join(parameter_names),
    )
    with client.begin() as con:
        con.exec_driver_sql(formatted_sql)
    return existing_udf(
        name=internal_name,
        input_types=in_types,
        output_type=out_type,
        schema=schema,
        parameters=parameter_names,
    )
