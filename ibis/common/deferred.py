from __future__ import annotations

from koerce import Deferred, Var


class _Variable(Var):
    def __repr__(self):
        return self.name


# reserved variable name for the value being matched
_ = Deferred(_Variable("_"))
