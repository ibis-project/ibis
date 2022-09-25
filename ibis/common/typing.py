import sys
from typing import Any, ForwardRef

from ibis.common.caching import memoize

if sys.version_info >= (3, 9):

    @memoize
    def evaluate_typehint(hint, module_name) -> Any:
        if isinstance(hint, str):
            hint = ForwardRef(hint)
        if isinstance(hint, ForwardRef):
            globalns = sys.modules[module_name].__dict__
            return hint._evaluate(globalns, locals(), frozenset())
        else:
            return hint

else:

    @memoize
    def evaluate_typehint(hint, module_name) -> Any:
        if isinstance(hint, str):
            hint = ForwardRef(hint)
        if isinstance(hint, ForwardRef):
            globalns = sys.modules[module_name].__dict__
            return hint._evaluate(globalns, locals())
        else:
            return hint
