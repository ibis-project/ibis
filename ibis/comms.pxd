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

from libc.stdint cimport (int8_t, int16_t, int32_t, int64_t,
                          uint8_t, uint16_t, uint32_t, uint64_t)

from posix.time cimport timespec, time_t
from posix.unistd cimport off_t

cdef extern from "ipc_support.h" nogil:

    int semarray_init(int size, unsigned short* init_vals)
    int semarray_exists(int sem_id)
    int semarray_delete(int sem_id)

    int semarray_lock(int sem_id, int i, timespec* timeout)
    int semarray_unlock(int sem_id, int i, timespec* timeout)


cdef extern from "sys/mman.h" nogil:
    enum:
        MAP_SHARED
        PROT_READ
        PROT_WRITE

    void* mmap(void* addr, size_t length, int prot, int flags, int fd,
               off_t offset)

    int munmap(void* addr, size_t length)

    enum:
        # SYNC and ASYNC are mutually exclusive
        MS_ASYNC
        MS_SYNC
        MS_INVALIDATE

    int msync(void* addr, size_t length, int flags)


cdef class BufferLike:
    """
    Superclass for data that is accessible in our virtual address space
    """
    cdef:
        int pos
        size_t size

    cdef readonly:
        bint closed

    cdef uint8_t* get_buffer(self) nogil


cdef class SharedMmap(BufferLike):

    cdef:
        int fd
        uint8_t* buf
        off_t offset
        object location
        object file_handle


cdef class IPCLock:
    cdef readonly:
        int semaphore_id

        # TODO: is more timeout granularity than milliseconds needed?
        int timeout_ms
        bint is_slave

    cdef create(self)
    cdef _check_semop_status(self, int ret)
    cdef inline int _our_sem_slot(self) nogil
    cdef inline int _their_sem_slot(self) nogil
    cdef void set_timeout(self, timespec* timeout) nogil


#----------------------------------------------------------------------
# Serialization / deserialization

ctypedef enum IbisCType:
    TYPE_NULL = 0
    TYPE_BOOLEAN = 1
    TYPE_TINYINT = 2
    TYPE_SMALLINT = 3
    TYPE_INT = 4
    TYPE_BIGINT = 5
    TYPE_FLOAT = 6
    TYPE_DOUBLE = 7
    TYPE_STRING = 8
    TYPE_VARCHAR = 9
    TYPE_CHAR = 10
    TYPE_TIMESTAMP = 11
    TYPE_DECIMAL = 12
    TYPE_FIXED_BUFFER = 13


cdef extern from *:
    ctypedef char* const_char_ptr "const char*"

cdef extern from "stdint.h" nogil:
    ctypedef int int32_t
    ctypedef int uint32_t

cdef extern from 'stdio.h' nogil:
    int snprintf(char *str, size_t size, char *format, ...)

cdef extern from "stdlib.h" nogil:
    void free(void * ptr)
    void * malloc(int size)

cdef extern from "string.h" nogil:
    void *memset(void *, int, size_t)

cdef extern from 'sys/time.h' nogil:
    ctypedef struct timeval:
        unsigned int tv_sec
        unsigned int tv_usec
    ctypedef int32_t time_t

# UUID Variant definitions
UUID_VARIANT_NCS = 0
UUID_VARIANT_DCE = 1
UUID_VARIANT_MICROSOFT = 2
UUID_VARIANT_OTHER = 3

# UUID Type definitions
UUID_TYPE_DCE_TIME = 1
UUID_TYPE_DCE_RANDOM = 4

cdef extern from 'uuid/uuid.h' nogil:
    ctypedef unsigned char uuid_t[16]

    void uuid_clear(uuid_t uu)

    int uuid_compare(uuid_t uu1, uuid_t uu2)

    void uuid_copy(uuid_t dst, uuid_t src)

    void uuid_generate(uuid_t out)
    void uuid_generate_random(uuid_t out)
    void uuid_generate_time(uuid_t out)

    int uuid_is_null(uuid_t uu)

    int uuid_parse(const_char_ptr indata, uuid_t uu)

    void uuid_unparse(uuid_t uu, char *out)
    void uuid_unparse_lower(uuid_t uu, char *out)
    void uuid_unparse_upper(uuid_t uu, char *out)

    time_t uuid_time(uuid_t uu, timeval *ret_tv)
    int uuid_type(uuid_t uu)
    int uuid_variant(uuid_t uu)
