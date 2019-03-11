"""
Selection execution functions for the arrow backend.
"""

import operator
from collections import OrderedDict
import numpy as np
from multipledispatch import Dispatcher
import pyarrow as pa
from pyarrow.lib import _is_primitive
from pyarrow.types import is_boolean
import toolz

import ibis.expr.operations as ops
import ibis.expr.types as ir
from ibis.compat import functools

from arrow.dispatch import execute_node
from arrow.core import execute
from arrow.execution import constants

compute_projection = Dispatcher(
    'compute_projection',
    doc="""\
Compute a projection, dispatching on whether we're computing a scalar, column,
or table expression.

Parameters
----------
expr : Union[ir.ScalarExpr, ir.ColumnExpr, ir.TableExpr]
parent : ops.Selection
data : pa.RecordBtach
scope : dict, optional

Returns
-------
value : scalar, pa.Array, pa.RecordBatch

Notes
-----
:class:`~ibis.expr.types.ScalarExpr` instances occur when a specific column
projection is a window operation.
""")


def remap_overlapping_column_names(table_op, root_table, data_columns):
    """Return an ``OrderedDict`` mapping possibly suffixed column names to
    column names without suffixes.

    Parameters
    ----------
    table_op : TableNode
        The ``TableNode`` we're selecting from.
    root_table : TableNode
        The root table of the expression we're selecting from.
    data_columns : set or frozenset
        The available columns to select from

    Returns
    -------
    mapping : OrderedDict[str, str]
        A map from possibly-suffixed column names to column names without
        suffixes.
    """
    if not isinstance(table_op, ops.Join):
        return None

    left_root, right_root = ops.distinct_roots(table_op.left, table_op.right)
    suffixes = {left_root: constants.LEFT_JOIN_SUFFIX,
                right_root: constants.RIGHT_JOIN_SUFFIX}
    column_names = [
        ({name, name + suffixes[root_table]} & data_columns, name)
        for name in root_table.schema.names
    ]
    mapping = OrderedDict(
        (toolz.first(col_name), final_name)
        for col_name, final_name in column_names if col_name
    )
    return mapping


def map_new_column_names_to_data(mapping, df):
    """Maps column names to data."""
    if mapping is not None:
        return df.loc[:, mapping.keys()].rename(columns=mapping)
    return df


def _compute_predicates(table_op, predicates, data, scope, **kwargs):
    """Compute the predicates for a table operation.

    Parameters
    ----------
    table_op : TableNode
    predicates : List[ir.ColumnExpr]
    data : pa.RecordBatch
    scope : dict
    kwargs : dict

    Returns
    -------
    computed_predicate : pa.Array[bool]

    Notes
    -----
    This handles the cases where the predicates are computed columns, in
    addition to the simple case of named columns coming directly from the input
    table.
    """
    for predicate in predicates:
        # Map each root table of the predicate to the data so that we compute
        # predicates on the result instead of any left or right tables if the
        # Selection is on a Join. Project data to only inlude columns from
        # the root table.
        root_tables = predicate.op().root_tables()

        # handle suffixes
        additional_scope = {}
        data_columns = frozenset(data.schema.names)

        for root_table in root_tables:
            mapping = remap_overlapping_column_names(
                table_op, root_table, data_columns
            )
            if mapping is not None:
                new_data = data.loc[:, mapping.keys()].rename(columns=mapping)
            else:
                new_data = data
            additional_scope[root_table] = new_data

        new_scope = toolz.merge(scope, additional_scope)
        yield execute(predicate, scope=new_scope, **kwargs)


@compute_projection.register(ir.ColumnExpr, pa.RecordBatch)
def compute_projection_column_expr(expr, data):
    """Computes projection."""
    op = expr.op()
    name = op.name
    return [data[get_index_by_name(data, name)], name]


def get_index_by_name(data, name):
    """Returns index of a dataframe with a specific name."""
    names = data.schema.names
    for i, column_name in enumerate(names):
        if column_name == name:
            return i
    raise LookupError("No column with name '{}'" % name)


@execute_node.register(ops.Selection, pa.RecordBatch)
def execute_selection_dataframe(op, data, scope=None, **kwargs):
    """Performs selection on a dataframe."""
    selections = op.selections
    predicates = op.predicates
    result = data
    if selections:
        data_pieces = []
        labels = []
        for selection in selections:
            arrow_array = compute_projection(
                selection,
                data,
            )
            data_pieces.append(arrow_array[0])
            labels.append(arrow_array[1])
        result = pa.RecordBatch.from_arrays(data_pieces, labels)

    # TODO: resolve scope initialization
    scope = {}

    if predicates:
        predicates = _compute_predicates(
            op.table.op(), predicates, data, scope, **kwargs
        )
        predicate = functools.reduce(operator.and_, predicates)
        assert len(predicate) == len(result), \
            'Selection predicate length does not match underlying table'
        result = loc(result, predicate)

    return result


def loc(record_batch, predicate):
    """Performs selection a record batch based on boolean values for each row."""
    # convert recordBatch to numpy ndarrays for effective computation
    data = to_numpy(record_batch)

    # get rows where predicate==true
    data = [pa.array(column[predicate]) for column in data]

    # return new record_batch object
    return pa.RecordBatch.from_arrays(data, record_batch.schema.names)


def to_numpy(record_batch):
    """
    Convert Arrow record_batch to an numpy.ndarray of ndarrays where each column of
    the record_batch is one numpy.ndarray

    Parameters
    ----------
    self: Arrow record_batch

    Returns
    -------
    numpyArray: numpy.ndarray
    """
    n = record_batch.num_columns
    numpy_array = np.empty(n, dtype=object)
    for i in range(n):

        column = record_batch.column(i)
        column_type = column.type
        if _is_primitive(column_type.id) \
                and not is_boolean(column_type):
            buflist = column.buffers()
            assert len(buflist) == 2
            offset = column.offset
            pandas_type = column_type.to_pandas_dtype()
            numpy_array[i] = np.frombuffer(buflist[-1],
                                           dtype=pandas_type)[offset:offset + len(column)]

        else:
            numpy_array[i] = column.to_pandas()
    return numpy_array
