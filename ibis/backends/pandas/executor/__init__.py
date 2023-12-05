from __future__ import annotations

import ibis.backends.pandas.executor.generic
import ibis.backends.pandas.executor.nested
import ibis.backends.pandas.executor.reductions
import ibis.backends.pandas.executor.relations
import ibis.backends.pandas.executor.strings
import ibis.backends.pandas.executor.temporal
import ibis.backends.pandas.executor.windows  # noqa: F401
from ibis.backends.pandas.executor.core import execute, zuper  # noqa: F401
