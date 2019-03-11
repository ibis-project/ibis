"""
Test for ArrowClient Class
"""
import unittest
import pyarrow as pa
import pandas as pd
import ibis.expr.types as ir
import ibis.expr.schema as sch

from arrow.client import client, ArrowClient


class TestClient(unittest.TestCase):
    def setUp(self):
        data = [pa.array([1, 2, 3, 4]), pa.array(['a', 'b', 'c', 'd'])]
        self.df = pa.RecordBatch.from_arrays(data, ['foo', 'bar'])
        self.dfs = {'t1': self.df}
        self.client = ArrowClient(self.dfs)

    def test__init__(self):
        self.assertTrue(isinstance(self.client, client.Client))

    def test_table(self):
        self.assertTrue(isinstance(self.client.table('t1'), ir.Expr))

    def test_schema(self):
        """
        test whether the schema of the PandasClient is equal to the schema of the ArrowClient

        Asserts:
            assertEqual: check schemas for equality
        """
        pandas_df = pd.DataFrame({'foo': [1, 2, 3, 4], 'bar': ['a', 'b', 'c', 'd']})
        pandas_schema = sch.infer(pandas_df, schema=None)
        arrow_schema = sch.infer(self.df, schema=None)
        self.assertEqual(pandas_schema, arrow_schema)


if __name__ == '__main__':
    unittest.main()
