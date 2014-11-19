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

# This module handles IPC coordination (using system semaphores) between Ibis
# workers and the Impala host, as well as low-level serialization and
# deserialization concerns via shared memory (memory maps).

# cython: embedsignature = True

from libc.errno cimport *

from comms cimport *

cdef class IPCLock:
    """
    Prototype class to deal with low-overhead control handoff between two
    processes on the same machine using system semaphores. By using two
    semaphores, we can implement a low-latency "my turn, your turn" approach
    that should not be prone to race conditions. Local testing suggests a
    handoff cycle could take less than 50 microseconds (in ideal system load
    conditions).

    Has a flag is_slave so that we can masquerade as the master (or slave) for
    testing purposes. When creating a non-slave, the "other" process lock
    starts off in unlocked state.
    """
    cdef readonly:
        int semaphore_id

        # TODO: is more timeout granularity than milliseconds needed?
        int timeout_ms
        bint is_slave

    def __cinit__(self, semaphore_id=None, int lock_timeout_ms=10,
                  bint is_slave=1):
        self.timeout_ms = lock_timeout_ms
        self.is_slave = is_slave

        if not is_slave:
            self.create()
            self.release()
        else:
            # This must not be None
            self.semaphore_id = semaphore_id

    def __dealloc__(self):
        # We need to deallocate the semaphore array if we created it
        if not self.is_slave:
            semarray_delete(self.semaphore_id)

    cdef create(self):
        # Initialize the semaphores and set in locked state.
        cdef unsigned short init_vals[2]
        init_vals[0] = 0
        init_vals[1] = 0
        self.semaphore_id = semarray_init(2, init_vals)

    def __repr__(self):
        # TODO: show lock state
        return 'IPCLock(semaphore_id=%d)' % self.semaphore_id

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, type, value, tb):
        self.release()

    def acquire(self, bint block=1):
        """
        Returns True on success, False on failure with timeout. Otherwise
        Exception is raised for other failures.
        """
        cdef:
            timespec timeout
            int ret

        # Try to lock this process's semaphore
        with nogil:
            if block:
                ret = semarray_lock(self.semaphore_id, self._our_sem_slot(),
                                    NULL)
            else:
                self.set_timeout(&timeout)
                ret = semarray_lock(self.semaphore_id,
                                    self._our_sem_slot(), &timeout)

        return self._check_semop_status(ret)

    def release(self, bint block=1):
        cdef:
            timespec timeout
            int ret

        # Try to unlock the other process's semaphore
        with nogil:
            if block:
                ret = semarray_unlock(self.semaphore_id,
                                      self._their_sem_slot(), NULL)
            else:
                self.set_timeout(&timeout)
                ret = semarray_unlock(self.semaphore_id,
                                      self._their_sem_slot(), &timeout)

        return self._check_semop_status(ret)

    cdef _check_semop_status(self, int ret):
        if ret == -1:
            if errno == EAGAIN:
                return False
            raise Exception('errno was %d' % ret)
        else:
            return True

    cdef inline int _our_sem_slot(self) nogil:
        return 0 if self.is_slave else 1

    cdef inline int _their_sem_slot(self) nogil:
        return 1 if self.is_slave else 0

    cdef void set_timeout(self, timespec* timeout) nogil:
        timeout.tv_sec = 0
        timeout.tv_nsec = self.timeout_ms * 1000000
