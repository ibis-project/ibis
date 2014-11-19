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

import threading
import unittest

import ibis.comms as comms


class TestIPCLock(unittest.TestCase):

    def setUp(self):
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


class TestMemoryMap(unittest.TestCase):
    pass
