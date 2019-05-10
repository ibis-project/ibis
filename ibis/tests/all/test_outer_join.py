import os
import unittest
import tempfile

from ibis.sql.sqlite.client import SQLiteClient


class TestFullOuterJoin(unittest.TestCase):
    def setUp(self):
        filehandle, path = tempfile.mkstemp()
        self.filehandle = filehandle
        self.path = path
        client = SQLiteClient(path, create=True)
        self.client = client

        client.con.execute("DROP TABLE IF EXISTS test_left;")
        client.con.execute("CREATE TABLE test_left(id_left integer, value_left integer);")
        client.con.execute("DROP TABLE IF EXISTS test_right;")
        client.con.execute("CREATE TABLE test_right(id_right integer, value_right integer);")

        self.left = client.table('test_left')
        self.right = client.table('test_right')

    def tearDown(self):
        self.client.con.dispose()
        os.close(self.filehandle)
        os.remove(self.path)

    def test_outer_join(self):
        left = self.left
        right = self.right
        joined = left.outer_join(right, left.id_left == right.id_right)
        sql_string = str(joined.compile())
        self.assertIn(
            'full outer',
            sql_string.lower()
        )


if __name__ == '__main__':
    unittest.main()
