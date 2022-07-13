import ibis
import ibis.expr.datatypes as dt
import ibis.expr.schema as sch
from ibis.common.validators import container_of
from ibis.expr import rules as rlz
from ibis.expr.operations.relations import PhysicalTable
from ibis.expr.operations.udf import UDFMixin


class TabularUserDefinedFunction(PhysicalTable, UDFMixin):
    input_type = container_of(rlz.datatype, type=tuple, max_length=0)

    @property
    def schema(self):
        if isinstance(self.return_type, dt.Struct):
            return sch.Schema(self.return_type.names, self.return_type.types)
        raise ValueError(f"Unknwon schema for return type {self.return_type}")

    def output_dtype(self):
        return ibis.table(self.schema())
