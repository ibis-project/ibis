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
import pytest

import pandas as pd

import ibis.compat as compat

from .test_comms import double_ex

from ibis.tasks import IbisTaskMessage, IbisTaskExecutor
from ibis.util import guid
from ibis.wire import BytesIO
import ibis.wire as wire

from ibis.compat import unittest
from ibis.tests.test_server import WorkerTestFixture

try:
    from ibis.comms import SharedMmap, IPCLock, IbisTableWriter
    SKIP_TESTS = False
except ImportError:
    SKIP_TESTS = True


pytestmark = pytest.mark.skipif(SKIP_TESTS or compat.PY3,
                                reason='Comms extension disabled')


class TestTasks(unittest.TestCase):

    def test_message_encode_decode(self):
        task = IbisTaskMessage(12345, 'foo', 12, 1000)

        encoded = task.encode()
        decoded = IbisTaskMessage.decode(encoded)
        encoded2 = decoded.encode()

        self.assertEqual(encoded, encoded2)

        attrs = ['semaphore_id', 'shmem_name', 'shmem_offset', 'shmem_size']
        for attr in attrs:
            self.assertEqual(getattr(task, attr), getattr(decoded, attr))


class TestPingPongTask(unittest.TestCase):

    def setUp(self):
        self.paths_to_delete = []

        # make size small so tracebacks get truncated
        path, size, offset = 'task_%s' % guid(), 36, 0
        self.paths_to_delete.append(path)

        self.lock = IPCLock(is_slave=0)
        self.task = IbisTaskMessage(self.lock.semaphore_id, path,
                                    offset, size)

        self.mm = SharedMmap(path, size, create=True)
        wire.PackedMessageWriter(self.mm).string('ping')

    def tearDown(self):
        for path in self.paths_to_delete:
            try:
                os.remove(path)
            except os.error:
                pass

    def test_execute_task(self):
        _execute_task(self.task, self.lock)

        self.mm.seek(0)
        reader = wire.PackedMessageReader(self.mm)
        assert reader.uint8()
        assert reader.string() == 'pong'


def _execute_task(task, master_lock):
    executor = IbisTaskExecutor(task)

    try:
        executor.execute()
    except:
        # Don't deadlock if execute has an exception
        executor.lock.release()
        raise

    # IPCLock has been released
    master_lock.acquire()


class TestTaskE2E(TestPingPongTask, WorkerTestFixture):

    def setUp(self):
        TestPingPongTask.setUp(self)
        WorkerTestFixture.setUp(self)

    def tearDown(self):
        TestPingPongTask.tearDown(self)
        WorkerTestFixture.tearDown(self)

    def test_task_end_to_end(self):
        msg = self._run_task(self.task)
        assert msg == 'ok'

        self.mm.seek(0)
        reader = wire.PackedMessageReader(self.mm)
        assert reader.uint8()
        assert reader.string() == 'pong'

    def test_task_return_exception(self):
        self.mm.seek(0)
        wire.PackedMessageWriter(self.mm).string('__unknown_task__')

        msg = self._run_task(self.task)
        assert msg == 'ok'

        self.mm.seek(0)
        reader = wire.PackedMessageReader(self.mm)
        assert reader.uint8() == 0
        error_msg = reader.string()
        assert 'Traceback' in error_msg

    def _run_task(self, task):
        encoded_task = task.encode()

        worker_port, worker_pid = self._spawn_worker()

        sock = self._connect(worker_port)
        sock.send(encoded_task)
        msg = sock.recv(1024)

        # IPCLock has been released
        self.lock.acquire()

        return msg


def delete_all_guid_files():
    import glob
    import os
    [os.remove(x) for x in glob.glob('*') if len(x) == 32]


class NRows(object):

    def __init__(self):
        self.total = 0

    def update(self, values):
        self.total += len(values)

    def merge(self, other):
        self.total += other.total
        return self

    def finalize(self):
        return self.total


class Summ(object):

    def __init__(self):
        self.total = 0

    def update(self, values):
        import pandas as pd
        self.total += pd.Series(values).sum()

    def merge(self, other):
        self.total += other.total
        return self

    def finalize(self):
        return self.total


class TestAggregateTasks(unittest.TestCase):

    def _get_mean_uda(self):
        # Dynamically generate the class instance. Use pandas so nulls are
        # excluded
        class Mean(object):

            def __init__(self):
                self.total = 0
                self.count = 0

            def update(self, values):
                values = pd.Series(values)
                self.total += values.sum()
                self.count += values.count()

            def merge(self, other):
                self.total += other.total
                self.count += other.count
                return self

            def finalize(self):
                return self.total / float(self.count)

        return Mean

    def setUp(self):
        self.paths_to_delete = []
        self.col_fragments = [double_ex(1000) for _ in range(10)]
        self.lock = IPCLock(is_slave=0)

    def test_update(self):
        klass = self._get_mean_uda()

        col = self.col_fragments[0]
        task, mm = self._make_update_task(klass, [col])

        _execute_task(task, self.lock)

        mm.seek(0)
        reader = wire.PackedMessageReader(mm)

        # success
        if not reader.uint8():
            raise Exception(reader.string())

        result = compat.pickle_load(reader.string())

        ex_total = pd.Series(col.to_numpy_for_pandas()).sum()
        assert result.total == ex_total

        # Test with prior state
        col = self.col_fragments[1]
        task, mm = self._make_update_task(klass, [col], prior_state=result)

        # Executor's turn again
        self.lock.release()
        _execute_task(task, self.lock)

        mm.seek(0)
        reader = wire.PackedMessageReader(mm)

        # success
        if not reader.uint8():
            raise Exception(reader.string())

        result = compat.pickle_load(reader.string())

        ex_total += pd.Series(col.to_numpy_for_pandas()).sum()

        # pandas will yield 0 on None input strangely
        assert ex_total != 0

        assert result.total == ex_total

    def test_merge(self):
        klass = self._get_mean_uda()

        lcol = self.col_fragments[0]
        rcol = self.col_fragments[1]

        left = self._update(klass, [lcol])
        right = self._update(klass, [rcol])

        task, mm = self._make_merge_task(left, right)
        _execute_task(task, self.lock)

        mm.seek(0)
        reader = wire.PackedMessageReader(mm)

        # success
        if not reader.uint8():
            raise Exception(reader.string())

        result = compat.pickle_load(reader.string())

        larr = lcol.to_numpy_for_pandas()
        rarr = rcol.to_numpy_for_pandas()
        assert larr is not None
        ex_total = (pd.Series(larr).sum() + pd.Series(rarr).sum())
        assert result.total == ex_total

    def test_finalize(self):
        klass = self._get_mean_uda()

        col = self.col_fragments[0]
        result = self._update(klass, [col])

        task, mm = self._make_finalize_task(result)
        _execute_task(task, self.lock)

        mm.seek(0)
        reader = wire.PackedMessageReader(mm)

        # success
        if not reader.uint8():
            raise Exception(reader.string())

        result = compat.pickle_load(reader.string())

        arr = col.to_numpy_for_pandas()
        ex_result = pd.Series(arr).mean()
        assert result == ex_result

    def _update(self, klass, args):
        task, mm = self._make_update_task(klass, args)
        _execute_task(task, self.lock)
        self.lock.release()

        mm.seek(0)
        reader = wire.PackedMessageReader(mm)

        # success
        if not reader.uint8():
            raise Exception(reader.string())

        return reader.string()

    def _make_update_task(self, uda_class, cols, prior_state=None):

        # Overall layout here:
        # - task name
        # - serialized agg class
        # - prior state flag 1/0
        # - (optional) serialized prior state
        # - serialized table fragment

        payload = BytesIO()
        msg_writer = wire.PackedMessageWriter(payload)
        msg_writer.string('agg-update')
        msg_writer.string(compat.pickle_dump(uda_class))

        if prior_state is not None:
            msg_writer.uint8(1)
            msg_writer.string(compat.pickle_dump(prior_state))
        else:
            msg_writer.uint8(0)

        writer = IbisTableWriter(cols)

        # Create memory map of the appropriate size
        path = 'task_%s' % guid()
        size = writer.total_size() + payload.tell()
        offset = 0
        mm = SharedMmap(path, size, create=True)
        self.paths_to_delete.append(path)

        mm.write(payload.getvalue())
        writer.write(mm)

        task = IbisTaskMessage(self.lock.semaphore_id, path, offset, size)

        return task, mm

    def _make_merge_task(self, left_pickled, right_pickled):
        payload = BytesIO()
        msg_writer = wire.PackedMessageWriter(payload)
        msg_writer.string('agg-merge')
        msg_writer.string(left_pickled)
        msg_writer.string(right_pickled)

        # Create memory map of the appropriate size
        path = 'task_%s' % guid()
        size = payload.tell()
        offset = 0
        mm = SharedMmap(path, size, create=True)
        self.paths_to_delete.append(path)

        mm.write(payload.getvalue())

        task = IbisTaskMessage(self.lock.semaphore_id, path, offset, size)

        return task, mm

    def _make_finalize_task(self, pickled):
        payload = BytesIO()
        msg_writer = wire.PackedMessageWriter(payload)
        msg_writer.string('agg-finalize')
        msg_writer.string(pickled)

        # Create memory map of the appropriate size
        path = 'task_%s' % guid()
        size = payload.tell()
        offset = 0
        mm = SharedMmap(path, size, create=True)
        self.paths_to_delete.append(path)

        mm.write(payload.getvalue())

        task = IbisTaskMessage(self.lock.semaphore_id, path, offset, size)

        return task, mm

    def tearDown(self):
        for path in self.paths_to_delete:
            try:
                os.remove(path)
            except os.error:
                pass
