from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow as pa

import ibis.expr.operations as ops
import ibis.expr.rules as rlz
import ibis.expr.schema as sch
from ibis import util
from ibis.common.grounds import Immutable

if TYPE_CHECKING:
    import pandas as pd


class PyArrowTableProxy(Immutable, util.ToFrame):
    __slots__ = ('_t', '_hash')

    def __init__(self, t: pa.Table) -> None:
        object.__setattr__(self, "_t", t)
        object.__setattr__(self, "_hash", hash((type(t), id(t))))

    def __hash__(self) -> int:
        return self._hash

    def __repr__(self) -> str:
        df_repr = util.indent(repr(self._t), spaces=2)
        return f"{self.__class__.__name__}:\n{df_repr}"

    def to_frame(self) -> pd.DataFrame:
        return self._t.to_pandas()

    def to_pyarrow(self, _: sch.Schema) -> pa.Table:
        return self._t


class PyArrowInMemoryTable(ops.InMemoryTable):
    data = rlz.instance_of(PyArrowTableProxy)


@sch.infer.register(pa.Table)
def infer_pyarrow_table_schema(t: pa.Table, schema=None):
    import ibis.backends.pyarrow.datatypes  # noqa: F401

    return sch.schema(schema if schema is not None else t.schema)
