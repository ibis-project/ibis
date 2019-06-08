import collections
import itertools

import sqlalchemy

import ibis.expr.rules as rlz
from ibis.expr.signature import Argument as Arg
from ibis.sql.postgres.compiler import PostgresUDFNode, add_operation


_udf_name_cache = collections.defaultdict(itertools.count)


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


def existing_udf(name, input_type, output_type, schema=None, parameter_names=None):
    """Create a ibis function that refers to an existing Postgres UDF already
    defined in database

    Parameters
    ----------
    name: str
    input_type : List[DataType]
    output_type : DataType
    schema: str - optionally specify the schema that the UDF is defined in
    parameter_names: str - give names to the arguments of the UDF

    Returns
    -------
    wrapper : Callable
        The wrapped function
    """
    if parameter_names is None:
        parameter_names = ['v{}'.format(i) for i in range(len(input_type))]
    else:
        assert len(input_type) == len(parameter_names)

    udf_node_fields = collections.OrderedDict([
        (name, Arg(rlz.value(type)))
        for name, type in zip(parameter_names, input_type)
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
