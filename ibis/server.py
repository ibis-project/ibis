# Daemon and worker server processes, heavily modified from PySpark (used as
# permitted under the Apache License 2.0)
#
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


import argparse
import errno
import numbers
import os
import select
import signal
import socket
import struct
import sys
import time
import threading
import traceback


from ibis.tasks import IbisTaskMessage, IbisTaskExecutor


SELECT_TIMEOUT = 0.25


def pack_uint32(val):
    return struct.pack('I', val)


class IbisTaskHandler(object):

    def __init__(self, sock):
        self.sock = sock

    def run(self):
        # TODO: task execution class should acquire OS semaphore, execute
        # task, then release the semaphore.

        encoded_task = self.sock.recv(1024)
        if not encoded_task:
            raise ValueError('Request was empty')

        try:
            task_msg = IbisTaskMessage.decode(encoded_task)
        except:
            self.sock.send(traceback.format_exc())
        else:
            # Acknowledge successful receipt
            self.sock.send('ok')
        finally:
            self.sock.close()

        self.execute(task_msg)

    def execute(self, task_msg):
        executor = IbisTaskExecutor(task_msg)
        return executor.execute()


def compute_real_exit_code(exit_code):
    # SystemExit's code can be integer or string, but os._exit only accepts
    # integers
    if isinstance(exit_code, numbers.Integral):
        return exit_code
    else:
        return 1


def _eintr_retry(func, *args):
    """restart a system call interrupted by EINTR"""
    while True:
        try:
            return func(*args)
        except (OSError, select.error) as e:
            if e.args[0] != errno.EINTR:
                raise

# ---------------------------------------------------------------------
# Daemon logic for spawning new child workers


class IbisServerNode(object):

    """
    This can be a daemon (for launching subprocesses) or a worker
    """

    def __init__(self, server_port=17001, daemon=True,
                 task_handler=IbisTaskHandler):
        self.server_port = server_port
        self.task_handler = task_handler

        self.setup_server_socket()

        # Can trigger shutdown to occur
        self._shutdown_request = False
        self._is_shutdown = threading.Event()
        self._is_shutdown.clear()

        self.is_daemon = daemon

        self.threaded_worker = False

        if self.is_daemon:
            # Create a new process group to corral our children
            os.setpgid(0, 0)
            self.set_daemon_signal_handlers()
        else:
            self.set_worker_signal_handlers()

    def set_daemon_signal_handlers(self):
        # Gracefully exit on SIGTERM
        signal.signal(signal.SIGTERM, self.handle_sigterm)

        # Don't die on SIGHUP
        signal.signal(signal.SIGHUP, signal.SIG_IGN)

    def handle_sigterm(self, *args):
        self.shutdown()

    def sighup_worker(self, *args):
        self._shutdown_request = True

    def sigint_worker(self, *args):
        # Terminate, with extreme prejudice
        self.listen_sock.close()
        self.shutdown()
        sys.exit(0)

    def set_worker_signal_handlers(self):
        signal.signal(signal.SIGHUP, self.sighup_worker)
        signal.signal(signal.SIGCHLD, signal.SIG_DFL)
        signal.signal(signal.SIGTERM, signal.SIG_DFL)

        # signal.signal(signal.SIGKILL, self.sigkill_worker)

        # restore the handler for SIGINT,
        # it's useful for debugging (show the stacktrace before exit)
        signal.signal(signal.SIGINT, self.sigint_worker)

    def shutdown(self):
        # Do we actually need this? Cannot be called when run from a thread
        # signal.signal(SIGTERM, SIG_DFL)

        if self.is_daemon:
            # Send SIGHUP to notify workers of shutdown
            os.kill(0, signal.SIGHUP)
        self._shutdown_request = True

    def setup_server_socket(self):
        # Bind the daemon socket listener
        self.listen_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.listen_sock.bind(('127.0.0.1', 0))
        self.listen_sock.listen(socket.SOMAXCONN)
        _, self.listen_port = self.listen_sock.getsockname()

    def report_daemon(self):
        print('Running daemon at port %d' % self.listen_port)
        self._server_send(pack_uint32(self.listen_port))

    def report_worker_info(self):
        print('Running worker at port %d' % self.listen_port)
        # Port and process id (as uint32)
        msg = struct.pack('II', self.listen_port, os.getpid())
        self._server_send(msg)

    def _server_send(self, msg):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(('localhost', self.server_port))
        sock.send(msg)

        data = sock.recv(1024)
        if data != 'ok':
            raise Exception('Server said: %s' % data)

    def run_daemon(self):
        self.report_daemon()

        while True:
            if self._shutdown_request:
                # shutdown was called
                self.listen_sock.close()
                self._is_shutdown.set()
                break

            ready_fds = _eintr_retry(select.select, [self.listen_sock],
                                     [], [], SELECT_TIMEOUT)[0]

            # cleanup in signal handler will cause deadlock
            self.cleanup_zombies()

            if self.listen_sock not in ready_fds:
                continue

            sock, _ = _eintr_retry(self.listen_sock.accept)
            msg = sock.recv(1024)

            if msg == 'new':
                try:
                    fork_ind = os.fork()
                except OSError as e:
                    if e.errno in (errno.EAGAIN, errno.EINTR):
                        time.sleep(1)
                        fork_ind = os.fork()  # error here will shutdown daemon
                    else:
                        # Signal that the fork failed
                        sock.send(pack_uint32(e.errno))
                        sock.close()
                        continue

                # Fork succeeded
                if fork_ind == 0:
                    self.become_worker(sock)
                else:
                    # Daemon process
                    sock.send('ok')
                    sock.close()
            elif msg == 'shutdown':
                self.shutdown()
                sock.send('ok')
                sock.close()
                continue
            elif msg.startswith('kill'):
                # e.g. kill 12345
                worker_pid = int(msg[4:])
                print('Killing %d' % worker_pid)
                try:
                    os.kill(worker_pid, signal.SIGINT)
                except OSError:
                    pass  # process already died
                sock.send('ok')
                sock.close()

    def cleanup_zombies(self):
        try:
            while True:
                pid, _ = os.waitpid(0, os.WNOHANG)
                if not pid:
                    break
        except:
            pass

    def become_worker(self, sock):
        """
        sock :
        """
        # in child process, close the server socket
        self.listen_sock.close()

        self.is_daemon = False

        # Setup a new socket server on some available port
        self.setup_server_socket()
        self.set_worker_signal_handlers()
        self.run_worker()

    def run_worker(self):
        self.report_worker_info()

        while True:
            if self._shutdown_request:
                # Triggered by SIGHUP
                self.listen_sock.close()
                self._is_shutdown.set()
                break

            # Await instructions
            ready_fds = _eintr_retry(select.select, [self.listen_sock],
                                     [], [], SELECT_TIMEOUT)[0]
            if not ready_fds:
                continue

            sock, _ = _eintr_retry(self.listen_sock.accept)

            task = self.task_handler(sock)

            if self.threaded_worker:
                # Spawn task in a daemon. These threads should not stay alive
                # if main worker thread exits. Revisit this at some point.
                t = threading.Thread(target=task.run)
                t.daemon = True
                t.start()
            else:
                # Run the task synchronously
                try:
                    task.run()
                except:
                    # Exception reporting is the task's job
                    pass


def parse_cl_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-D', '--daemon', dest='is_daemon',
                        default=True, action='store_true')

    parser.add_argument('-p', '--port', type=int, dest='impala_port',
                        default=17001, action='store')

    return parser.parse_known_args()[0]


if __name__ == '__main__':
    args = parse_cl_args()

    if args.is_daemon:
        node = IbisServerNode(args.impala_port)
        try:
            node.run_daemon()
        except SystemExit:
            pass
        except:
            node.shutdown()
            traceback.print_exc()
    else:
        raise NotImplementedError
