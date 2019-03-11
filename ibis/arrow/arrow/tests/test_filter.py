"""
Unittest for filter operations
"""
import unittest
from datetime import datetime as dtime
import ibis
import pyarrow as pa
import numpy as np
import pandas as pd

from arrow.client import ArrowClient

class Filter(unittest.TestCase):
    """Test class for filtering on integers and strings
    for small and large datasets
    """
    def setUp(self):
        # simple
        times = pd.Series([dtime(2019, 1, 5),
                          dtime(2019, 1, 6),
                          dtime(2019, 1, 7),
                          dtime(2020, 1, 5)])
        data = [pa.array([1, 2, 3, 4]),
                pa.array(['a', 'b', 'c', 'd']),
                pa.array(times)
                ]
        df = pa.RecordBatch.from_arrays(data, ['foo', 'bar', 'baz'])
        dfs = {'t1': df}

        client = ArrowClient(dfs)
        self.table = client.table('t1')

        self.expected_result0 = pd.DataFrame({'foo': [3, 4],
                                              'bar': ['c', 'd'],
                                              'baz': [dtime(2019, 1, 7),
                                                      dtime(2020, 1, 5)],
                                              },
                                             index=[0, 1])
        self.expected_result1 = pd.DataFrame({'foo': [1, 2],
                                              'bar': ['a', 'b'],
                                              'baz': [dtime(2019, 1, 5),
                                                      dtime(2019, 1, 6)],
                                              },
                                             index=[0, 1])
        self.expected_result2 = pd.DataFrame({'foo': [2, 3, 4],
                                              'bar': ['b', 'c', 'd'],
                                              'baz': [dtime(2019, 1, 6),
                                                      dtime(2019, 1, 7),
                                                      dtime(2020, 1, 5)],
                                              },
                                             index=[0, 1, 2])
        self.expected_result3 = pd.DataFrame({'foo': [1, 2, 3],
                                              'bar': ['a', 'b', 'c'],
                                              'baz': [dtime(2019, 1, 5),
                                                      dtime(2019, 1, 6),
                                                      dtime(2019, 1, 7)],
                                              },
                                             index=[0, 1, 2])
        self.expected_result4 = pd.DataFrame({'foo': [4],
                                              'bar': ['d'],
                                              'baz': [dtime(2020, 1, 5)],
                                              },
                                             index=[0])
        self.expected_result5 = pd.DataFrame({'foo': [2, 3],
                                              'bar': ['b', 'c'],
                                              'baz': [dtime(2019, 1, 6),
                                                      dtime(2019, 1, 7)],
                                              },
                                             index=[0, 1])
        self.expected_result6 = pd.DataFrame({'foo': [2],
                                              'bar': ['b'],
                                              'baz': [dtime(2019, 1, 6)],
                                              },
                                             index=[0])

        # big
        # test parameters
        self.low = -100
        self.high = 100
        column_length = 100
        num_columns = 100

        # create data for test__big()
        pyarrow_data = []
        pandas_data = {}
        for i in range(num_columns):
            array = np.random.rand(column_length)
            array = np.add(self.low, np.multiply(array, self.high - self.low))

            pyarrow_data.append(pa.array(array))
            pandas_data["col" + str(i)] = array
        pyarrow_names = ["col" + str(i) for i in range(num_columns)]

        # create data frames
        pyarrow_df = pa.RecordBatch.from_arrays(pyarrow_data, pyarrow_names)
        pyarrow_dfs = {"df1": pyarrow_df}

        pandas_df = pd.DataFrame(pandas_data)
        pandas_dfs = {"df1": pandas_df}

        # extract table
        pyarrow_client = ArrowClient(pyarrow_dfs)
        self.pyarrow_table = pyarrow_client.table("df1")

        pandas_client = ibis.pandas.connect(pandas_dfs)
        self.pandas_table = pandas_client.table("df1")


    def test__filter(self):
        """Tests filtering for constant integer values and constant strings
        Additionally tests concatenation of predicates with boolean operators
        """
        # tests filtering for integer columns
        result1 = self.table.filter([self.table.foo > 2]).execute()
        self.assertTrue(result1.to_pandas().equals(self.expected_result0))

        result2 = self.table.filter([self.table.foo < 3]).execute()
        self.assertTrue(result2.to_pandas().equals(self.expected_result1))

        result3 = self.table.filter([self.table.foo >= 2]).execute()
        self.assertTrue(result3.to_pandas().equals(self.expected_result2))

        result4 = self.table.filter([self.table.foo <= 3]).execute()
        self.assertTrue(result4.to_pandas().equals(self.expected_result3))

        result5 = self.table.filter([self.table.foo == 4]).execute()
        self.assertTrue(result5.to_pandas().equals(self.expected_result4))

        # tests filtering for string columns
        result6 = self.table.filter([self.table.bar > 'b']).execute()
        self.assertTrue(result6.to_pandas().equals(self.expected_result0))

        result7 = self.table.filter([self.table.bar < 'c']).execute()
        self.assertTrue(result7.to_pandas().equals(self.expected_result1))

        result8 = self.table.filter([self.table.bar >= 'b']).execute()
        self.assertTrue(result8.to_pandas().equals(self.expected_result2))

        result9 = self.table.filter([self.table.bar <= 'c']).execute()
        self.assertTrue(result9.to_pandas().equals(self.expected_result3))

        result10 = self.table.filter([self.table.bar == 'd']).execute()
        self.assertTrue(result10.to_pandas().equals(self.expected_result4))

        # tests filtering with boolean operators and mixed types
        result16 = self.table.filter([self.table.foo >= 2,
                                      self.table.bar <= 'c']).execute()
        self.assertTrue(result16.to_pandas().equals(self.expected_result5))

        result17 = self.table.filter([(self.table.foo >= 2)
                                      & (self.table.bar <= 'c')]).execute()
        self.assertTrue(result17.to_pandas().equals(self.expected_result5))

        result18 = self.table.filter([(self.table.foo == 2)
                                      & (self.table.bar == 'b')]).execute()
        self.assertTrue(result18.to_pandas().equals(self.expected_result6))

        result19 = self.table.filter([(self.table.foo == 2)
                                      | (self.table.bar == 'c')]).execute()
        self.assertTrue(result19.to_pandas().equals(self.expected_result5))

        result20 = self.table.filter([((self.table.foo == 2)
                                       | (self.table.foo == 3))
                                      & (self.table.bar == 'b')
                                      & (self.table.foo != 4)]).execute()
        self.assertTrue(result20.to_pandas().equals(self.expected_result6))

    def test__filter_big(self):
        """Tests filtering on big tables created from random values
        """
        threshold1 = self.high / 2
        threshold2 = self.high / 4

        pyarrow_con1 = [self.pyarrow_table.col1 > threshold1]
        pandas_con1 = [self.pandas_table.col1 > threshold1]
        pyarrow_result1 = self.pyarrow_table.filter(pyarrow_con1).execute()
        pandas_result1 = self.pandas_table.filter(pandas_con1).execute()
        self.assertTrue(pyarrow_result1.to_pandas().equals(pandas_result1))

        pyarrow_con2 = [self.pyarrow_table.col1 < threshold1]
        pandas_con2 = [self.pandas_table.col1 < threshold1]
        pyarrow_result2 = self.pyarrow_table.filter(pyarrow_con2).execute()
        pandas_result2 = self.pandas_table.filter(pandas_con2).execute()
        self.assertTrue(pyarrow_result2.to_pandas().equals(pandas_result2))

        pyarrow_con3 = [(self.pyarrow_table.col1 < threshold1),
                        (self.pyarrow_table.col1 != threshold2)]
        pandas_con3 = [(self.pandas_table.col1 < threshold1),
                       (self.pandas_table.col1 != threshold2)]
        pyarrow_result3 = self.pyarrow_table.filter(pyarrow_con3).execute()
        pandas_result3 = self.pandas_table.filter(pandas_con3).execute()
        self.assertTrue(pyarrow_result3.to_pandas().equals(pandas_result3))

        pyarrow_con4 = [(self.pyarrow_table.col1 > threshold1)
                        & (self.pyarrow_table.col2 <= threshold2)]
        pandas_con4 = [(self.pandas_table.col1 > threshold1)
                       & (self.pandas_table.col2 <= threshold2)]
        pyarrow_result4 = self.pyarrow_table.filter(pyarrow_con4).execute()
        pandas_result4 = self.pandas_table.filter(pandas_con4).execute()
        self.assertTrue(pyarrow_result4.to_pandas().equals(pandas_result4))

        pyarrow_con5 = [(self.pyarrow_table.col1 > threshold1)
                        | (self.pyarrow_table.col1 < threshold2)]
        pandas_con5 = [(self.pandas_table.col1 > threshold1)
                       | (self.pandas_table.col1 < threshold2)]
        pyarrow_result5 = self.pyarrow_table.filter(pyarrow_con5).execute()
        pandas_result5 = self.pandas_table.filter(pandas_con5).execute()
        self.assertTrue(pyarrow_result5.to_pandas().equals(pandas_result5))


if __name__ == '__main__':
    unittest.main()
