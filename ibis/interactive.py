import ibis
import ibis.examples as ex
from ibis import deferred as _
from ibis import selectors as s
from ibis.expr.operations import udf

ibis.options.interactive = True

__all__ = ["_", "ex", "ibis", "s", "udf"]
