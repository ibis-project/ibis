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

from ibis.backends.base import Database
from ibis.backends.base.sql.alchemy import BaseAlchemyBackend

from .client import SQLiteClient


class Backend(BaseAlchemyBackend):
    name = 'sqlite'
    client_class = SQLiteClient
    # TODO check if there is a reason to not use the parent AlchemyDatabase, or
    # if there is technical debt that makes this required
    database_class = Database

    def connect(self, path=None, create=False):

        """
        Create an Ibis client connected to a SQLite database.

        Multiple database files can be created using the attach() method

        Parameters
        ----------
        path : string, default None
            File path to the SQLite database file. If None, creates an
            in-memory transient database and you can use attach() to add more
            files
        create : boolean, default False
            If file does not exist, create it
        """
        self.client = SQLiteClient(backend=self, path=path, create=create)
        return self.client

    def list_tables(self, like=None, database=None):
        if database is None:
            database = self.current_database
        return super().list_tables(like, database=database)
