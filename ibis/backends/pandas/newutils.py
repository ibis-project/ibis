from __future__ import annotations

import itertools
import operator
from collections.abc import Sized

import numpy as np
import pandas as pd


def asframe(values):
    if isinstance(values, dict):
        names, values = zip(*values.items())
    elif isinstance(values, (list, tuple)):
        names = [f"_{i}" for i in range(len(values))]
    else:
        raise TypeError(f"values must be a dict, list, or tuple; got {type(values)}")

    size = 1
    all_scalars = True
    for v in values:
        if isinstance(v, pd.Series):
            size = len(v)
            all_scalars = False
            break

    columns = []
    for v in values:
        if isinstance(v, pd.Series):
            pass
        elif isinstance(v, (list, np.ndarray)):
            v = pd.Series(itertools.repeat(np.array(v), size))
        else:
            v = pd.Series(np.repeat(v, size))
        columns.append(v)

    return pd.concat(columns, axis=1, keys=names), all_scalars


def rowwise(func, values, **kwargs):
    # dealing with a collection of series objects
    df, all_scalars = asframe(values)
    result = df.apply(func, axis=1, **kwargs)
    return result.iloc[0] if all_scalars else result


def elementwise(func, values, **kwargs):
    if isinstance(values, pd.Series):
        # dealing with a single series object
        return values.apply(func, **kwargs)
    else:
        # dealing with a single scalar object
        return func(values)
