import operator

import toolz

# pandas compat
try:
    from pandas.api.types import (  # noqa: F401
        DatetimeTZDtype,
        CategoricalDtype,
        infer_dtype
    )
except ImportError:
    from pandas.types.dtypes import (  # noqa: F401
        DatetimeTZDtype,
        CategoricalDtype,
        infer_dtype
    )

try:
    from pandas.core.tools.datetimes import to_time, to_datetime  # noqa: F401
except ImportError:
    from pandas.tseries.tools import to_time, to_datetime  # noqa: F401


to_date = toolz.compose(operator.methodcaller('date'), to_datetime)
