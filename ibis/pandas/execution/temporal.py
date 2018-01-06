import itertools
import operator

import six

import regex as re

import numpy as np
import pandas as pd

import toolz

from pandas.core.groupby import SeriesGroupBy

import ibis

from ibis.compat import reduce, maketrans
import ibis.expr.operations as ops

from ibis.pandas.dispatch import execute_node
from ibis.pandas.core import integer_types, scalar_types


@execute_node.register(ops.Date, pd.Series)
def execute_timestamp_date(op, data, **kwargs):
    return data.dt.floor('d')


@execute_node.register(ops.TimestampTruncate, pd.Series)
def execute_timestamp_truncate(op, data, **kwargs):
    return data.dt.floor(op.unit)
