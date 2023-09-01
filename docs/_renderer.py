from __future__ import annotations

import griffe.dataclasses as dc  # noqa: F401
import griffe.docstrings.dataclasses as ds  # noqa: F401
import griffe.expressions as expr  # noqa: F401
import quartodoc.ast as qast
from plum import dispatch
from quartodoc import MdRenderer


class Renderer(MdRenderer):
    style = "ibis"

    @dispatch
    def render(self, el: qast.ExampleCode):
        return f"""```
{el.value}
```"""
