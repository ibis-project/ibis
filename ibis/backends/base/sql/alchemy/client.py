from typing import Dict

import sqlalchemy as sa

import ibis.expr.schema as sch
from ibis.backends.base.sql import SQLClient


class AlchemyClient(SQLClient):
    def __init__(self, con: sa.engine.Engine) -> None:
        super().__init__()
        self.backend.con = con
        self.backend.meta = sa.MetaData(bind=con)
        self.backend._inspector = sa.inspect(con)
        self.backend._schemas: Dict[str, sch.Schema] = {}
