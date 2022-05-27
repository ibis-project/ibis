from ibis.expr.datatypes.core import *  # noqa: F401,F403
from ibis.expr.datatypes.core import _normalize, _WellKnownText  # noqa: F401

halffloat = float16  # noqa: F405
float = float64  # noqa: F405
double = float64  # noqa: F405
int = int64  # noqa: F405
uint_ = uint64  # noqa: F405
bool = boolean  # noqa: F405
str = string  # noqa: F405
bytes = binary  # noqa: F405

validate_type = dtype  # noqa: F405
