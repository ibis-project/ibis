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

from .client import SQLiteDatabase


def compile(expr):
    """
    Force compilation of expression for the SQLite target
    """
    from .client import SQLiteDialect
    from ibis.sql.alchemy import to_sqlalchemy
    return to_sqlalchemy(expr, dialect=SQLiteDialect)


def connect(path):
    return SQLiteDatabase(path)
