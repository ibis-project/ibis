from __future__ import annotations


import pandas as pd

import ibis.expr.operations as ops
from ibis.backends.pandas.newpandas import execute
from ibis.backends.pandas.newutils import columnwise, rowwise
import numpy as np

@execute.register(ops.E)
def execute_e(op):
    return np.e


@execute.register(ops.Pi)
def execute_pi(op):
    return np.pi
