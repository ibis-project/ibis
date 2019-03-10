"""
Defines the ibid internal ArrowTable and ArrowClient classes.
"""
import ibis.expr.operations as ops
import ibis.expr.datatypes as dt
import ibis.expr.types as ir
import ibis.expr.schema as sch
import ibis.client as client
import pyarrow as pa
import six


@sch.infer.register(pa.lib.RecordBatch)
def infer_pyarrow_schema(df, schema=None):
    """Creates a schema based on a record batch."""
    schema = schema if schema is not None else {}

    pairs = []
    for column_name, arrow_dtype in zip([item.name for item in df.schema],
                                        [item.type for item in df.schema]):
        if not isinstance(column_name, six.string_types):
            raise TypeError(
                'Column names must be strings to use the pandas backend'
            )

        if column_name in schema:
            ibis_dtype = dt.dtype(schema[column_name])
        else:
            ibis_dtype = dt.dtype(arrow_dtype)
        pairs.append((column_name, ibis_dtype))

    return sch.schema(pairs)


class ArrowTable(ops.DatabaseTable):
    """Defines the ibis internal ArrowTable."""
    pass


class ArrowClient(client.Client):
    """
        Client gives API functionality for tables
        :function __init__: initialize the ArrowClient with a dictionary
        :function table: returns expression object of table
    """
    def __init__(self, dictionary):
        self.dictionary = dictionary

    def table(self, name, schema=None):
        """
        :param name: name of the table to select
        :return: selected table
        """
        df = self.dictionary[name]
        schema = sch.infer(df, schema=schema)
        return ArrowTable(name, schema, self).to_expr()

    def execute(self, query, params=None, limit='default', **kwargs):
        """
        Executes a expression on this ArrowClient.
        :param query: the expression
        :param params: the parameters
        :param limit: the limit
        :param kwargs:
        :return:
        """
        # params limit und **kwargs argumente are required because the
        # function call happens in the ibis backend and can not be altered
        from arrow.core import execute

        if limit != 'default':
            raise ValueError(
                'limit parameter to execute is not yet implemented in the '
                'arrow backend'
            )

        assert isinstance(query, ir.Expr)
        return execute(query)
