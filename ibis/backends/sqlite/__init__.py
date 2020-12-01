# Copyright 2015 Cloudera Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import ibis.backends.base
from ibis.backends.base_sqlalchemy.alchemy import AlchemyQueryBuilder

from .client import SQLiteClient
from .compiler import SQLiteDialect


class Backend(ibis.backends.base.BaseBackend):
    name = 'sqlite'
    builder = AlchemyQueryBuilder
    dialect = SQLiteDialect

    def connect(self):
        return SQLiteClient(path, create=create)
