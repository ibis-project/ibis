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

import pytest

import os
import socket
import struct
import threading

from ibis.compat import unittest
from ibis.server import IbisServerNode
import ibis.compat as compat


# non-POSIX system (e.g. Windows)
pytestmark = pytest.mark.skipif(compat.PY3 or not hasattr(os, 'setpgid'),
                                reason='non-POSIX system')


try:
    import psutil
    HAVE_PSUTIL = True
except ImportError:
    HAVE_PSUTIL = False


def get_proc(pid):
    import psutil
    return psutil.Process(pid)


def port_is_closed(port):
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        server_sock.bind(('127.0.0.1', port))
    except socket.error:
        return False
    return True


class ImpalaServerFixture(object):

    def setUp(self):
        # This takes the place of the Impala server socket for testing
        self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_sock.bind(('127.0.0.1', 0))
        self.server_sock.listen(socket.SOMAXCONN)
        _, self.server_port = self.server_sock.getsockname()

    def tearDown(self):
        self.server_sock.close()

    def _acknowledge_process(self):
        sock, _ = self.server_sock.accept()
        msg = sock.recv(1024)
        sock.send('ok')
        sock.close()
        return msg


class TestDaemon(unittest.TestCase, ImpalaServerFixture):

    def setUp(self):
        ImpalaServerFixture.setUp(self)

    def tearDown(self):
        ImpalaServerFixture.tearDown(self)

    def test_shutdown_closes_port(self):
        daemon = IbisServerNode(server_port=self.server_port)

        t = threading.Thread(target=daemon.run_daemon)
        t.start()

        # The daemon won't actually start its main loop until the server
        # acknowledges its existence
        self._acknowledge_process()

        # Request shutdown and wait for thread to finish
        daemon.shutdown()
        t.join()
        assert port_is_closed(daemon.listen_port)

    def test_daemon_report_port(self):
        daemon = IbisServerNode(server_port=self.server_port)

        exceptions = []

        def f():
            try:
                daemon.report_daemon()
            except Exception as e:
                exceptions.append(str(e))

        t = threading.Thread(target=f)
        t.start()
        self._acknowledge_process()
        t.join()
        self.assertEqual(len(exceptions), 0)

        t = threading.Thread(target=f)
        t.start()
        sock, _ = self.server_sock.accept()
        sock.recv(1024)
        sock.send('not ok')
        sock.close()
        t.join()
        self.assertEqual(len(exceptions), 1)


class WorkerTestFixture(ImpalaServerFixture):

    def setUp(self):
        ImpalaServerFixture.setUp(self)

        self.daemon = IbisServerNode(server_port=self.server_port)

        self.daemon_t = threading.Thread(target=self.daemon.run_daemon)
        self.daemon_t.start()
        self._acknowledge_process()

    def tearDown(self):
        ImpalaServerFixture.tearDown(self)

        # Request shutdown and wait for thread to finish
        if self.daemon_t.isAlive():
            self.daemon.shutdown()
            self.daemon_t.join()

    def _connect(self, port=None):
        if port is None:
            port = self.daemon.listen_port
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(('localhost', port))
        return sock

    def _spawn_worker(self):
        if not HAVE_PSUTIL:
            pytest.skip('no psutil')

        sock = self._connect()

        # Ask to create a worker; reply OK on successful fork
        sock.send('new')
        reply = sock.recv(1024)
        assert reply == 'ok'
        sock.close()

        # Acknowledge the worker's existence
        sock, _ = self.server_sock.accept()
        msg = sock.recv(1024)
        sock.send('ok')
        sock.close()

        worker_port, worker_pid = struct.unpack('II', msg)
        proc = get_proc(worker_pid)
        assert proc.status != ('running', 'sleeping')
        return worker_port, worker_pid


class TestWorkerManagement(WorkerTestFixture, unittest.TestCase):

    def test_spawn_worker_and_kill(self):
        worker_port, worker_pid = self._spawn_worker()

        # Kill the worker
        sock = self._connect()
        sock.send('kill %d' % worker_pid)
        msg = sock.recv(1024)
        assert msg == 'ok'

        # This could hang forever if the process isn't dying
        os.waitpid(worker_pid, 0)

        with pytest.raises(psutil.NoSuchProcess):
            psutil.Process(worker_pid).status()

        # Check that the worker port is now closed
        assert port_is_closed(worker_port)

    def test_daemon_request_shutdown(self):
        worker_port, worker_pid = self._spawn_worker()
        worker_port2, worker_pid2 = self._spawn_worker()

        # Workers really are alive
        assert not port_is_closed(worker_port)
        assert not port_is_closed(worker_port2)

        sock = self._connect()
        sock.send('shutdown')
        reply = sock.recv(1024)
        assert reply == 'ok'

        os.waitpid(worker_pid, 0)
        os.waitpid(worker_pid2, 0)
        with pytest.raises(psutil.NoSuchProcess):
            psutil.Process(worker_pid).status()

        # Check that the worker port is now closed
        assert port_is_closed(worker_port)
        assert port_is_closed(worker_port2)

        self.daemon_t.join()
        assert not self.daemon_t.isAlive()
