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

import traceback

from ibis.wire import PackedMessageReader, PackedMessageWriter
import ibis.comms as comms
import ibis.wire as wire


class IbisTaskMessage(object):
    """
    Prototype wire protocol for task descriptions

    uint32_t semaphore_id
    uint32_t shmem_name_len
    char* shmem_name
    uint64_t shmem_offset
    uint64_t shmem_size
    """
    def __init__(self, semaphore_id, shmem_name, shmem_offset, shmem_size):
        self.semaphore_id = semaphore_id
        self.shmem_name = shmem_name
        self.shmem_offset = shmem_offset
        self.shmem_size = shmem_size

    @classmethod
    def decode(self, message):
        """
        Convert from the bytestring wire protocol

        Parameters
        ----------
        message : bytes

        Returns
        -------
        message : IbisTaskMessage
        """
        buf = PackedMessageReader(message)
        sem_id = buf.uint32()
        shmem_name = buf.string()
        shmem_offset = buf.uint64()
        shmem_size = buf.uint64()
        return IbisTaskMessage(sem_id, shmem_name, shmem_offset, shmem_size)

    def encode(self):
        """
        Format this message as a bytestring according to the current version of
        the wire protocol.

        Returns
        -------
        encoded : bytes

        """
        buf = PackedMessageWriter()
        buf.uint32(self.semaphore_id)
        buf.string(self.shmem_name)
        buf.uint64(self.shmem_offset)
        buf.uint64(self.shmem_size)
        return buf.get_result()


class Task(object):
    """
    Prototype

    Run task in a thread, capture tracebacks or other problems.
    """
    def __init__(self, shmem):
        shmem.seek(0)
        self.shmem = shmem
        self.complete = False

    def mark_success(self):
        wire.write_uint8(self.shmem, 1)

    def mark_failure(self):
        wire.write_uint8(self.shmem, 0)

    def execute(self):
        pass

    def run(self):
        raise NotImplementedError

    def done(self):
        pass


_task_registry = {}

def register_task(kind, task_class, override=False):
    """
    Register a new task implementation with the execution system
    """
    if kind in _task_registry and not override:
        raise KeyError('Task of type %s is already defined and '
                       'override is False')

    _task_registry[kind] = task_class


class IbisTaskExecutor(object):
    """
    Runs the requested task and handles locking, exception reporting, and so
    forth.
    """
    def __init__(self, task_msg):
        self.task_msg = task_msg

        self.lock = comms.IPCLock(self.task_msg.semaphore_id)
        self.shmem = comms.SharedMmap(self.task_msg.shmem_name,
                                      self.task_msg.shmem_size,
                                      offset=self.task_msg.shmem_offset)

    def _cycle_ipc_lock(self):
        # TODO: I put this here as a failsafe in case the task needs to bail
        # out for a known reason and we want to immediately release control to
        # the master process
        self.lock.acquire()
        self.lock.release()

    def execute(self):
        # TODO: Timeout concerns
        self.lock.acquire()

        # TODO: this can block forever on malformed input
        task_type = PackedMessageReader(self.shmem).string()

        try:
            klass = _task_registry[task_type]
            task = klass(self.shmem)
            task.run()
        except:
            self.shmem.seek(0)

            # XXX: Failure indicator
            wire.write_uint8(self.shmem, 0)

            tb = traceback.format_exc()

            # HACK: Traceback string must be truncated so it will fit in the
            # shared memory (along with the uint32 length prefix)
            if len(tb) + 5 > len(self.shmem):
                tb = tb[:len(self.shmem) - 5]

            wire.write_string(self.shmem, tb)
        finally:
            self.lock.release()


#----------------------------------------------------------------------
# Ping pong task for testing


class PingPongTask(Task):

    def __init__(self, shmem):
        Task.__init__(self, shmem)

    def run(self):
        self.mark_success()
        wire.write_string(self.shmem, 'pong')


register_task('ping', PingPongTask)

#----------------------------------------------------------------------
# Aggregation execution tasks


class AggregationUpdateTask(Task):
    """

    """
    def run(self):
        # Shared memory includes the serialized aggregation class (as a string)
        # followed by
        pass


class AggregationMergeTask(Task):

    def run(self):
        # Objects to merge stored in length-prefixed strings in shared memory
        pass


class AggregationFinalizeTask(Task):

    def run(self):
        # Single length-prefixed string to finalize
        pass
