# Copyright 2014 Cloudera Inc.
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

import os
import sys
import threading

import pytest

import numpy as np

from ibis.util import guid
from ibis.compat import unittest

try:
    import ibis.comms as comms
    from ibis.comms import (SharedMmap, IbisType, IbisTableReader,
                            IbisTableWriter)
    SKIP_TESTS = False
except ImportError:
    SKIP_TESTS = True


def _nuke(path):
    try:
        os.remove(path)
    except os.error:
        pass

pytestmark = pytest.mark.skipif(SKIP_TESTS,
                                reason='Comms extension disabled')


class TestIPCLock(unittest.TestCase):

    def setUp(self):
        if sys.platform == 'darwin':
            raise unittest.SkipTest

        self.timeout = 1
        self.master = comms.IPCLock(is_slave=0, lock_timeout_ms=self.timeout)
        self.slave = comms.IPCLock(self.master.semaphore_id,
                                   lock_timeout_ms=self.timeout)

    def test_acquire_and_release(self):
        # It's not our turn
        self.assertFalse(self.master.acquire(block=False))

        self.slave.acquire()
        self.slave.release()

        self.assertTrue(self.master.acquire())

    def test_cleanup_semaphore_arrays(self):
        # Otherwise, there will be too many semaphore arrays floating around
        for i in range(500):
            comms.IPCLock(is_slave=0)

    def test_thread_blocking(self):
        lock = threading.Lock()

        results = []

        # This also verifies that the GIL is correctly dropped
        def ping():
            while True:
                with self.slave:
                    with lock:
                        if len(results) == 4:
                            break
                        results.append('ping')

        def pong():
            while True:
                with self.master:
                    with lock:
                        if len(results) == 4:
                            break
                        results.append('pong')

        t1 = threading.Thread(target=pong)
        t1.start()

        t2 = threading.Thread(target=ping)
        t2.start()

        t1.join()
        t2.join()

        ex_results = ['ping', 'pong'] * 2
        assert results == ex_results


class TestSharedMmap(unittest.TestCase):

    def setUp(self):
        self.to_nuke = []

    def tearDown(self):
        for path in self.to_nuke:
            _nuke(path)

    def test_create_file(self):
        size = 1024

        path = guid()
        try:
            mm = SharedMmap(path, size, create=True)
            mm.close()

            self.assertTrue(os.path.exists(path))
            self.assertEqual(os.stat(path).st_size, size)
        finally:
            _nuke(path)

    def test_file_not_exist(self):
        path = guid()
        self.assertRaises(IOError, SharedMmap, path, 1024)
        self.assertRaises(IOError, SharedMmap, path, 1024, offset=20,
                          create=True)

    def test_close_file(self):
        path = guid()
        self.to_nuke.append(path)
        data = guid()

        mm = SharedMmap(path, len(data), create=True)
        assert mm.closed is False
        mm.close()
        assert mm.closed is True

        # idempotent
        mm.close()
        assert mm.closed is True

        self.assertRaises(IOError, mm.read, 4)
        self.assertRaises(IOError, mm.write, 'bazqux')
        self.assertRaises(IOError, mm.seek, 0)
        self.assertRaises(IOError, mm.flush)

    def test_file_interface(self):
        path = guid()
        self.to_nuke.append(path)
        data = guid()

        mm = SharedMmap(path, len(data), create=True)

        assert mm.tell() == 0
        mm.write(data)
        assert mm.tell() == len(data)

        mm.seek(0)
        assert mm.tell() == 0
        result = mm.read(16)
        assert len(result) == 16
        assert result == data[:16]
        assert mm.tell() == 16

    def test_multiple_mmaps(self):
        path = guid()
        path2 = guid()
        data = guid()
        self.to_nuke.extend([path, path2])

        mm1 = SharedMmap(path, len(data), create=True)
        mm1.write(data)

        mm2 = SharedMmap(path, len(data))
        result = mm2.read()
        self.assertEqual(result, data)

        # Open both maps first, see if data synchronizes
        mm1 = SharedMmap(path2, len(data), create=True)
        mm2 = SharedMmap(path2, len(data))

        mm1.write(data)
        result = mm2.read()
        self.assertEqual(result, data)


def rand_bool(N):
    return np.random.randint(0, 2, size=N).astype(np.uint8)


def rand_int_span(dtype, N):
    info = np.iinfo(dtype)
    lo, hi = info.min, info.max
    return np.random.randint(lo, hi, size=N).astype(dtype)


def bool_ex(N):
    mask = rand_bool(N)
    values = rand_bool(N)
    return _to_masked(values, mask, IbisType.BOOLEAN)


def int_ex(N, ibis_type):
    mask = rand_bool(N)

    nptype = comms._ibis_to_numpy[ibis_type]
    values = rand_int_span(nptype, N)
    return _to_masked(values, mask, ibis_type)


def double_ex(N):
    mask = rand_bool(N)
    values = np.random.randn(N)
    return _to_masked(values, mask, IbisType.DOUBLE)


def _to_masked(values, mask, dtype):
    return comms.masked_from_numpy(values, mask, dtype)


class TestImpalaMaskedFormat(unittest.TestCase):

    """
    Check that data makes it to and from the file format, and that it can be
    correctly transformed to the appropriate NumPy/pandas/etc. format
    """
    N = 1000

    def _check_roundtrip(self, columns):
        writer = IbisTableWriter(columns)

        table_size = writer.total_size()

        buf = comms.RAMBuffer(table_size)
        writer.write(buf)
        buf.seek(0)
        reader = IbisTableReader(buf)

        for i, expected in enumerate(columns):
            result = reader.get_column(i)

            assert result.equals(expected)

    def test_basic_diverse_table(self):
        columns = [
            bool_ex(self.N),
            int_ex(self.N, IbisType.TINYINT),
            int_ex(self.N, IbisType.SMALLINT),
            int_ex(self.N, IbisType.INT),
            int_ex(self.N, IbisType.BIGINT)
        ]
        self._check_roundtrip(columns)

    def test_boolean(self):
        col = bool_ex(self.N)
        self.assertEqual(col.nbytes(), self.N * 2)
        self._check_roundtrip([col])

        # Booleans with nulls will come out as object arrays with None for each
        # null. This is how pandas handles things
        result = col.to_numpy_for_pandas()
        assert result.dtype == object
        _check_masked_correct(col, result, np.bool_,
                              lambda x: x is None)

        # No nulls, get boolean dtype
        mask = np.zeros(self.N, dtype=np.uint8)
        values = rand_bool(self.N)
        col2 = _to_masked(values, mask, IbisType.BOOLEAN)
        result2 = col2.to_numpy_for_pandas()
        _check_masked_correct(col2, result2, np.bool_,
                              lambda x: x is None)

        # Get a numpy.ma.MaskedArray
        # masked_result = col.to_masked_array()

        # didn't copy
        # assert not masked_result.flags.owndata
        # assert masked_result.base is col

    # For each integer type, address conversion back to NumPy rep's: masked
    # array, pandas-compatible (nulls force upcast to float + NaN for NULL)
    def test_tinyint(self):
        col = int_ex(self.N, IbisType.TINYINT)
        self.assertEqual(col.nbytes(), self.N * 2)
        self._check_roundtrip([col])

        _check_pandas_ints_nulls(col, np.int8)

        _check_pandas_ints_no_nulls(self.N, IbisType.TINYINT)

    def test_smallint(self):
        col = int_ex(self.N, IbisType.SMALLINT)
        self.assertEqual(col.nbytes(), self.N * 3)
        self._check_roundtrip([col])

        _check_pandas_ints_nulls(col, np.int16)

        _check_pandas_ints_no_nulls(self.N, IbisType.SMALLINT)

    def test_int(self):
        col = int_ex(self.N, IbisType.INT)
        self.assertEqual(col.nbytes(), self.N * 5)
        self._check_roundtrip([col])

        _check_pandas_ints_nulls(col, np.int32)

        _check_pandas_ints_no_nulls(self.N, IbisType.INT)

    def test_int_segfault(self):
        col = int_ex(1000000, IbisType.INT)
        col.to_numpy_for_pandas()

    def test_bigint(self):
        col = int_ex(self.N, IbisType.BIGINT)
        self.assertEqual(col.nbytes(), self.N * 9)
        self._check_roundtrip([col])

        _check_pandas_ints_nulls(col, np.int64)

        _check_pandas_ints_no_nulls(self.N, IbisType.BIGINT)

    def test_float(self):
        mask = rand_bool(self.N)
        values = np.random.randn(self.N).astype(np.float32)
        col = _to_masked(values, mask, IbisType.FLOAT)
        self.assertEqual(col.nbytes(), self.N * 5)
        self._check_roundtrip([col])

        result = col.to_numpy_for_pandas()
        assert result.dtype == np.float32
        mask = np.isnan(result)
        ex_mask = col.mask().view(np.bool_)
        assert np.array_equal(mask, ex_mask)

    def test_double(self):
        col = double_ex(self.N)
        self.assertEqual(col.nbytes(), self.N * 9)
        self._check_roundtrip([col])

        result = col.to_numpy_for_pandas()
        assert result.dtype == np.float64
        mask = np.isnan(result)
        ex_mask = col.mask().view(np.bool_)
        assert np.array_equal(mask, ex_mask)

    def test_string_pyobject(self):
        # pandas handles strings in object-type (NPY_OBJECT) arrays and uses
        # either None or NaN for nulls. For the time being we'll be consistent
        # with that
        #
        pass

    def test_timestamp(self):
        pass

    def test_decimal(self):
        pass

    def test_multiple_string_columns(self):
        # For the time being, string (STRING, VARCHAR, CHAR) columns will all
        # share the same intern table
        pass


def _check_pandas_ints_nulls(col, dtype):
    result = col.to_numpy_for_pandas()
    assert result.dtype == np.float64
    _check_masked_correct(col, result, dtype, np.isnan)


def _check_pandas_ints_no_nulls(N, ibis_type):
    nptype = comms._ibis_to_numpy[ibis_type]

    mask = np.zeros(N, dtype=np.uint8)
    values = rand_int_span(nptype, N)
    col = _to_masked(values, mask, ibis_type)

    result = col.to_numpy_for_pandas()
    assert result.dtype == nptype
    _check_masked_correct(col, result, nptype, lambda x: False)


def _check_masked_correct(col, result, dtype, is_na_f):
    mask = col.mask()
    data = col.data_bytes().view(dtype)
    for i, v in enumerate(result):
        if mask[i]:
            assert is_na_f(v)
        else:
            # For comparisons outside representable integer range, this may
            # yield incorrect results
            assert v == data[i]


class TestTableRoundTrip(unittest.TestCase):

    """
    Test things not captured by datatype-specific tests
    """

    def test_table_metadata(self):
        # Check values from preamble
        pass
