from __future__ import annotations

import sys
from typing import Any, ForwardRef

import toolz

# TODO(kszucs): try to use inspect.get_annotations() backport instead

if sys.version_info >= (3, 9):

    @toolz.memoize
    def evaluate_typehint(hint, module_name=None) -> Any:
        if isinstance(hint, str):
            hint = ForwardRef(hint)
        if isinstance(hint, ForwardRef):
            if module_name is None:
                globalns = {}
            else:
                globalns = sys.modules[module_name].__dict__
            return hint._evaluate(globalns, locals(), frozenset())
        else:
            return hint

else:

    @toolz.memoize
    def evaluate_typehint(hint, module_name) -> Any:
        if isinstance(hint, str):
            hint = ForwardRef(hint)
        if isinstance(hint, ForwardRef):
            if module_name is None:
                globalns = {}
            else:
                globalns = sys.modules[module_name].__dict__
            return hint._evaluate(globalns, locals())
        else:
            return hint
