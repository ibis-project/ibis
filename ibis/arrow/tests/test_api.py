"""
Tests for general IBIS Api calls
"""
import unittest
import pyarrow as pa

from arrow.api import connect



class TestAPI(unittest.TestCase):
    def test_connect(self):
        """
        Creates dummy DataFrame and establishes connection to the Arrow Client
        """
        data = [pa.array([1, 2, 3, 4]), pa.array(['a', 'b', 'c', 'd'])]
        batch = pa.RecordBatch.from_arrays(data, ['foo', 'bar'])
        dictionary = {"t1": batch}

        con = connect(dictionary)
        self.assertEqual(con.dictionary, dictionary)


if __name__ == '__main__':
    unittest.main()
