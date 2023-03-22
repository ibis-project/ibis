from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow as pa

import ibis.expr.schema as sch
from ibis.expr.operations.relations import TableProxy

if TYPE_CHECKING:
    import pandas as pd


class PyArrowTableProxy(TableProxy):
    __slots__ = ()

    def to_frame(self) -> pd.DataFrame:
        return self._data.to_pandas()

    def to_pyarrow(self, _: sch.Schema) -> pa.Table:
        return self._data


@sch.infer.register(pa.Table)
def infer_pyarrow_table_schema(t: pa.Table, schema=None):
    import ibis.backends.pyarrow.datatypes  # noqa: F401

    return sch.schema(schema if schema is not None else t.schema)
