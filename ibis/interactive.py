import ibis
from ibis import deferred as _
from ibis.expr import selectors as s

ibis.options.interactive = True

__all__ = ["ibis", "_", "s"]
