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
import unittest

from ibis.tasks import IbisTaskMessage, IbisTaskExecutor
from ibis.comms import SharedMmap, IPCLock
from ibis.util import guid
import ibis.wire as wire

from ibis.tests.test_server import WorkerTestFixture


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
        executor = IbisTaskExecutor(self.task)

        try:
            executor.execute()
        except:
            # Don't deadlock if execute has an exception
            executor.lock.release()
            raise

        # IPCLock has been released
        self.lock.acquire()

        self.mm.seek(0)
        reader = wire.PackedMessageReader(self.mm)
        assert reader.uint8()
        assert reader.string() == 'pong'


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



class TestAggregateTasks(unittest.TestCase):
    pass
