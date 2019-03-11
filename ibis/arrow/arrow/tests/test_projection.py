"""
Unittest for projection on columns
"""
import unittest
import pyarrow as pa
import pandas as pd
from arrow.client import ArrowClient


class Projection(unittest.TestCase):
    def setUp(self):
        """Initialization of example data
        """
        data = [pa.array([1, 2, 3, 4]),
                pa.array(['a', 'b', 'c', 'd']),
                pa.array([5, 6, 7, 8])
               ]
        df = pa.RecordBatch.from_arrays(data, ['foo', 'bar', 'baz'])
        dfs = {'t1': df}

        client = ArrowClient(dfs)
        self.table = client.table('t1')

        self.expected_result0 = pd.DataFrame({'foo': [1, 2, 3, 4]})
        self.expected_result1 = pd.DataFrame({'foo': [1, 2, 3, 4],
                                              'bar': ['a', 'b', 'c', 'd'],
                                              'baz': [5, 6, 7, 8]
                                              })
        self.expected_result2 = pd.DataFrame({'foo': [1, 2, 3, 4],
                                              'bar': ['a', 'b', 'c', 'd']
                                              })

    def test__projection(self):
        result0 = self.table.projection(["foo"]).execute()
        self.assertTrue(result0.to_pandas().equals(self.expected_result0))

        result1 = self.table.projection([]).execute()
        self.assertTrue(result1.to_pandas().equals(self.expected_result1))

        result2 = self.table.projection(["foo", "bar"]).execute()
        self.assertTrue(result2.to_pandas().equals(self.expected_result2))


if __name__ == '__main__':
    unittest.main()
