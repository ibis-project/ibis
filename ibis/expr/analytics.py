import warnings

from ibis.expr.operations.histograms import *  # noqa: F401,F403

warnings.warn("ibis.expr.analytics will be removed in ibis 3.0", FutureWarning)

del warnings
