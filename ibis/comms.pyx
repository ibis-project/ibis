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
# cython: boundscheck = False
# cython: wraparound = False

from libc.errno cimport *
from libc.stdlib cimport free, malloc, realloc
from libc.string cimport memcpy, memcmp

from comms cimport *

from numpy cimport ndarray
cimport numpy as cnp

cimport cpython as cp
from cython.operator cimport (dereference as deref,
                              preincrement as preinc,
                              predecrement as predec)
cimport cython

import numpy as np
import os

cnp.import_array()

cdef double NaN = <double> np.NaN

cdef fused signed_int:
    int8_t
    int16_t
    int32_t
    int64_t


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

    def __cinit__(self, semaphore_id=None, int lock_timeout_ms=1,
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


cdef class BufferLike:

    def __len__(self):
        return self.size

    cdef uint8_t* get_buffer(self) nogil:
        return NULL

    def read(self, int nbytes=-1):
        self._raise_if_closed()

        if (nbytes < 0) or nbytes > (self.size - self.pos):
            # Read no more than the remaining capacity of the buffer
            nbytes = self.size - self.pos

        result = cp.PyBytes_FromStringAndSize(<char*> self.get_buffer(),
                                              nbytes)

        self.pos += nbytes

        return result

    def write(self, object s):
        """
        Write UTF8 encoded bytes to the memory map
        """
        self._raise_if_closed()

        if not isinstance(s, bytes):
            s = s.encode('utf-8')

        cdef char* s_bytes = cp.PyBytes_AsString(s)
        cdef size_t length = len(s)
        with nogil:
            memcpy(self.get_buffer(), s_bytes, length)

        self.pos += length

    def seek(self, int where):
        self._raise_if_closed()

        if where < 0 or where >= self.size:
            raise IOError('Position out of bounds')

        self.pos = where

    def _raise_if_closed(self):
        # No-op
        pass


cdef class RAMBuffer(BufferLike):
    cdef:
        uint8_t* buf

    def __cinit__(self, size):
        self.size = size
        self.buf = <uint8_t*> malloc(self.size)
        self.pos = 0

    def __dealloc__(self):
        if self.buf != NULL:
            free(self.buf)

    cdef uint8_t* get_buffer(self) nogil:
        return self.buf + self.pos


cdef class SharedMmap(BufferLike):
    """
    Simple interface to shared memory exchanged between Ibis and the parent
    process via memory maps. It also implements the file interface so that it
    can be treated as a file-like object by pure Python code.
    """

    def __cinit__(self, location, size, offset=0, create=False):
        self.location = location
        self.size = size
        self.offset = offset

        if not os.path.exists(self.location):
            if not create:
                # Don't create the file if it's not there already
                raise IOError('%s does not exist' % self.location)
            elif offset != 0:
                raise IOError('File does not exist; nonzero offset invalid')

            # Create the file and truncate to indicated size
            self.file_handle = open(self.location, 'wb+')
            self.file_handle.truncate(size)
        else:
            self.file_handle = open(self.location, 'rb+')

        # Memory-map the file, raise on failure
        cdef int fd = self.file_handle.fileno()
        with nogil:
            self.buf = <uint8_t*> mmap(NULL, self.size, PROT_READ | PROT_WRITE,
                                       MAP_SHARED, fd, self.offset)

        if self.buf == NULL:
            # TODO: check various errno failure conditions, raise a more
            # informative exception
            raise Exception('Memory mapping failed')

        self.closed = 0

        # So we can treat this as file-like
        self.pos = 0

    def __dealloc__(self):
        self.close()

    cdef uint8_t* get_buffer(self) nogil:
        return self.buf + self.pos

    def __repr__(self):
        return ('SharedMmap(%s, size=%d, offset=%d)' %
                (self.location, self.size, self.offset))

    def close(self):
        if self.closed:
            return

        with nogil:
            munmap(<void*> self.buf, self.size)
        try:
            self.file_handle.close()
        except:
            pass

        self.closed = 1

    def tell(self):
        self._raise_if_closed()
        return self.pos

    def flush(self):
        cdef int ret

        self._raise_if_closed()

        with nogil:
           ret = msync(<void*> self.buf, self.size, MS_SYNC)
        if ret == -1:
            raise IOError('msync failed')

    def _raise_if_closed(self):
        if self.closed:
            raise IOError('File is closed')


#----------------------------------------------------------------------
# Read and write tables with as little copying as possible (preferably nearly
# zero in the case of primitive types) from the binary format delivered by
# Impala.
#
# TODO: is endianness ever a concern?
# TODO: do we care about the column names? Adds complexity
#
# OMITTED FOR NOW
# uint32_t*
#   name offsets (K + 1 of them)
# char*
#   column names (no null terminators)
#
# Version 1 of binary table layout
#
# uint32_t
#   M magic number
# uint32_t
#   K number of columns
# uint64_t
#   table length
# uint8_t*
#   T column dtype codes
# uint32_t*
#   O byte offsets (K + 1 offsets, so we have an exact length for each)
# uint8_t
#   F Column format
# COLUMN* array of column blocks
# INTERN_TABLE
#   All strings refer to a master array of strings at the end of the blob
#
# The most general column format looks like the following:
# uint8_t*
#   null mask
# void*
#   data
#
# I'm putting a format code in the table layout to leave the door open for
# delivery formats that are friendlier to the receiving data structure. For
# example: badger format arrays (especially strings / category types) can be
# received with less copying than either pandas/NumPy containers.
#
# One problem with this is that it will require copying for most practical
# input formats (NumPy, pandas, badger, etc.) in use by users. Any string data
# used by NumPy or pandas users will have to be copied / boxed in Python
# objects, anyway. We also will have to adopt pandas conventions for missing
# data in boolean and integer arrays (casting to a different NumPy dtype and
# using a missing value marker).
#
# String columns: uint32_t* with references into master intern table
# Intern table format (all UTF8 bytes, no null terminators):
#   uint32_t K (length)
#   uint32_t* offsets (K + 1 size)
#   char* data
#
#   So offsets[0] = 0, and offsets[K] is the total size of the data array. We
#   don't store any nulls at all.
#
# Timestamps and decimals will require some specialized handling.

# Keep around instances of the common numeric dtypes for simplicity
NPY_U1 = np.dtype('u1')
NPY_U2 = np.dtype('u2')
NPY_U4 = np.dtype('u4')
NPY_U8 = np.dtype('u8')
NPY_I1 = np.dtype('i1')
NPY_I2 = np.dtype('i2')
NPY_I4 = np.dtype('i4')
NPY_I8 = np.dtype('i8')
NPY_F4 = np.dtype('f4')
NPY_F8 = np.dtype('f8')
NPY_O = np.dtype('O')


cdef uint32_t IMPALA_MAGIC_UINT32 = 1337959792

# Only implementing the first type for now
cdef uint8_t FORMAT_MASKED = 0
cdef uint8_t FORMAT_PANDAS = 1

# cdef uint8_t FORMAT_BADGER = 2


# Enum-like class, for use on the Python side
cdef class IbisType:
    BOOLEAN = TYPE_BOOLEAN
    TINYINT = TYPE_TINYINT
    SMALLINT = TYPE_SMALLINT
    INT = TYPE_INT
    BIGINT = TYPE_BIGINT
    FLOAT = TYPE_FLOAT
    DOUBLE = TYPE_DOUBLE
    STRING = TYPE_STRING

    # TODO: Are these all handled as strings in Ibis-land?
    VARCHAR = TYPE_VARCHAR
    CHAR = TYPE_CHAR

    TIMESTAMP = TYPE_TIMESTAMP
    DECIMAL = TYPE_DECIMAL


_ibis_to_numpy = {
    IbisType.BOOLEAN: NPY_U1,
    IbisType.TINYINT: NPY_I1,
    IbisType.SMALLINT: NPY_I2,
    IbisType.INT: NPY_I4,
    IbisType.BIGINT: NPY_I8,
    IbisType.FLOAT: NPY_F4,
    IbisType.DOUBLE: NPY_F8,
    IbisType.STRING: NPY_O,
    IbisType.CHAR: NPY_O,
    IbisType.VARCHAR: NPY_O,
    IbisType.TIMESTAMP: NPY_I8,
    IbisType.DECIMAL: NPY_F8  # HACK
}

_ibis_stride = {
    IbisType.BOOLEAN: 1,
    IbisType.TINYINT: 1,
    IbisType.SMALLINT: 2,
    IbisType.INT: 4,
    IbisType.BIGINT: 8,
    IbisType.FLOAT: 4,
    IbisType.DOUBLE: 8,
    IbisType.STRING: 1
}

cdef inline int type_to_stride(int dtype) except -1:
    return _ibis_stride[dtype]


cdef check_numpy_compat(ndarray arr, ibis_type):
    ex_dtype = _ibis_to_numpy[ibis_type]
    if arr.dtype != ex_dtype:
        raise TypeError('Needed type to be %s, was %s' %
                        (str(ex_dtype), str(arr.dtype)))


_type_names = {
    TYPE_BOOLEAN: 'boolean'
}


cdef class InternTableBuilder:
    """
    Quick and dirty so we can build our own intern tables for testing purposes
    """
    cdef:
        char* data
        uint32_t* offsets
        uint32_t length
        uint32_t data_cap
        uint32_t offsets_cap

        # hash table
        int* ht
        int hash_size

    def __cinit__(self):
        self.offsets = self.data = NULL

        self.length = 0

        self.offsets_cap = 1024
        self.offsets = <uint32_t*> malloc(self.offsets_cap * 4)

        if self.offsets == NULL:
            raise MemoryError

        self.data_cap = 32768
        self.data = <char*> malloc(self.data_cap)

        if self.data == NULL:
            free(self.offsets)
            raise MemoryError

    def __dealloc__(self):
        if self.offsets != NULL:
            free(self.offsets)

        if self.data != NULL:
            free(self.data)

    cdef inline uint32_t get(self, char* val, size_t length):
        """
        Get code for string, and add to intern table if not in there already,
        """
        pass

    def finalize(self):
        """
        Save the current state of the builder as an immutable InternTable
        """


# Fowler-Noll-Vo hash function for strings, ported from Impala
cdef uint64_t FNV64_PRIME = 1099511628211UL
cdef inline uint64_t fnv_hash64(void* data, int bytes, uint64_t hash):
    cdef uint8_t* ptr = <uint8_t*> data
    while bytes > 0:
        hash = (deref(ptr) ^ hash) * FNV64_PRIME
        preinc(ptr)
        predec(bytes)
    return hash

cdef inline uint32_t fnv_hash32(void* data, int bytes, uint32_t hash):
    cdef uint64_t hash_u64 = hash | (<uint64_t>hash << 32)
    hash_u64 = fnv_hash64(data, bytes, hash_u64)
    return (hash_u64 >> 32) ^ (hash_u64 & 0xFFFFFFFF)


cdef class InternTable:
    """

    """
    cdef:
        int fmt
        uint32_t* offsets
        uint8_t* data
        size_t length

        # hash table
        uint32_t* ht

    def __cinit__(self, format='pybytes'):
        # TODO: the intern table might not necessarily want to produce PyBytes
        # objects in all cases (e.g. if we have a string container that can
        # handle raw bytes)
        self.fmt = 0

    cdef nbytes(self):
        return self.offsets[self.length]

    cdef init(self, uint32_t* offsets, uint8_t* data, size_t length):
        self.data = data
        self.offsets = offsets
        self.length = length

    cdef write_buffer(self, uint8_t* buf):
        """
        Write the table to the passed buffer
        """
        cdef BufferInterface face = BufferInterface()
        face.set_buffer(buf)

        face.write_array(self.offsets, self.length + 1, 4)
        face.write_array(self.data, self.nbytes(), 1)

    cdef inline uint32_t get(self, char* val, size_t length):
        pass


cdef class IbisTableReader:
    """

    """
    cdef readonly:
        int ncolumns
        uint64_t length

    cdef:
        uint8_t* buf
        size_t bufsize

        uint8_t* dtypes
        uint32_t* col_offsets
        uint8_t table_format

        uint8_t* data_start
        InternTable intern_table

    def __cinit__(self, BufferLike container, format='numpy'):
        self.buf = container.get_buffer()
        self.bufsize = container.size

        cdef BufferInterface reader = BufferInterface()
        reader.set_buffer(self.buf)

        cdef uint32_t magic = reader.read_uint32()
        if magic != IMPALA_MAGIC_UINT32:
            raise ValueError('Magic code at start of buffer did not match')

        self.ncolumns = reader.read_uint32()
        self.length = reader.read_uint64()

        cdef uint8_t* data = self.buf + reader.pos

        # TODO: decide if column names are desired

        # Just the bytes there. Any reason to copy?
        self.dtypes = data
        data += self.ncolumns

        self.col_offsets = <uint32_t*> data
        data += 4 * (self.ncolumns + 1)

        # Validate this?
        self.table_format = data[0]

        self.data_start = data + 1

        # Initialize the intern table
        self.intern_table = None

    def get_column(self, i):
        # TODO: boundscheck
        if self.table_format == FORMAT_MASKED:
            return self._read_masked(i)
        else:
            raise NotImplementedError

    cdef _read_masked(self, int i):
        cdef MaskedColumnReader reader = MaskedColumnReader()

        reader.init(self.dtypes[i], self.length,
                    self.data_start + self.col_offsets[i],
                    self.intern_table)

        return reader.read()

    cdef _read_pandas(self, int i):
        raise NotImplementedError


cdef class IbisColumnReader:
    pass



cdef class MaskedColumnReader(IbisColumnReader):

    cdef:
        uint8_t* buf
        uint8_t dtype
        uint64_t length
        InternTable intern_table

    cdef init(self, uint8_t dtype, uint64_t length, uint8_t* buf,
              InternTable table):
        self.buf = buf
        self.dtype = dtype
        self.length = length
        self.intern_table = table

    def read(self):
        cdef MaskedColumn result = MaskedColumn()

        result.dtype = self.dtype
        result.length = self.length
        result.stride = type_to_stride(self.dtype)

        result.null_mask = self.buf
        result.data = self.buf + self.length

        return result

#----------------------------------------------------------------------
# Column data handlers and coercion to various compatible formats

cdef class IbisColumn:

    cdef readonly:
        int dtype
        uint64_t length
        size_t stride

    # N.B. all Cython cdef methods are "virtual" in the C++ sense, so it's safe
    # to use cdef IbisColumn and you'll get the subclass methods
    cpdef nbytes(self):
        raise NotImplementedError

    cdef write_buffer(self, uint8_t* buf):
        raise NotImplementedError


cdef class MaskedColumn(IbisColumn):
    """
    Adapter class for masked array format data with columnar layout (null mask
    and data stored in contiguous arrays).
    """
    cdef:
        # We never own this data
        uint8_t* null_mask
        uint8_t* data

        # In case we need to hold on to references to some objects
        list obj_refs

    format_code = FORMAT_MASKED

    cpdef nbytes(self):
        # The number of bytes taken up by the table in binary-serialized form
        return self.length * (self.stride + 1)

    def __len__(self):
        return self.length

    cdef init_from_buffer(self, uint8_t* buf, int dtype,
                          uint64_t length, size_t stride):
        self.dtype = dtype
        self.length = length
        self.stride = stride

        self.null_mask = buf
        self.data = buf + length

    cdef write_buffer(self, uint8_t* buf):
        memcpy(buf, self.null_mask, self.length)
        memcpy(buf + self.length, self.data, self.length * self.stride)

    def mask(self):
        return buffer_to_numpy_view(self.null_mask, self.length, cnp.NPY_UINT8)

    def data_bytes(self):
        return buffer_to_numpy_view(self.data, self.length * self.stride,
                                    cnp.NPY_UINT8)

    def to_numpy_for_pandas(self, copy=False):
        """
        Produce a new array (copy of data) containing data in a suitable
        representation for immediate use in pandas.

        Parameters
        ----------
        copy : bool, default False
            Avoid copying any data if we can

        Returns
        -------
        arr : ndarray
        """
        # TODO: reduce code duplication
        if self.dtype == TYPE_BOOLEAN:
            return _box_pandas_bool(self.data, self.null_mask, self.length,
                                    copy=copy)
        elif self.dtype == TYPE_TINYINT:
            return _box_pandas_integer(<int8_t*> self.data, self.null_mask,
                                       self.length, NPY_I1, copy=copy)
        elif self.dtype == TYPE_SMALLINT:
            return _box_pandas_integer(<int16_t*> self.data, self.null_mask,
                                       self.length, NPY_I2, copy=copy)
        elif self.dtype == TYPE_INT:
            return _box_pandas_integer(<int32_t*> self.data, self.null_mask,
                                       self.length, NPY_I4, copy=copy)
        elif self.dtype == TYPE_BIGINT:
            return _box_pandas_integer(<int64_t*> self.data, self.null_mask,
                                       self.length, NPY_I8, copy=copy)
        elif self.dtype == TYPE_FLOAT:
            return _box_pandas_floating(<float*> self.data, self.null_mask,
                                        self.length, NPY_F4, copy=copy)
        elif self.dtype == TYPE_DOUBLE:
            return _box_pandas_floating(<double*> self.data, self.null_mask,
                                        self.length, NPY_F8, copy=copy)
        elif self.dtype == TYPE_TIMESTAMP:
            raise NotImplementedError
        elif self.dtype == TYPE_DECIMAL:
            raise NotImplementedError

    def to_masked_array(self, copy=False):
        """
        Create a numpy.ma.MaskedArray from the, and do not copy the data if
        possible by default

        Parameters
        ----------
        copy: bool, default False
            Avoid copying any data if we can

        Returns
        -------
        arr : numpy.ma.MaskedArray
        """
        pass

    def equals(self, MaskedColumn other):
        if (self.dtype != other.dtype or
            self.length != other.length):
            return False

        # Compare the data ignoring data behind the null mask
        cdef int i

        for i in range(self.length):
            if self.null_mask[i] != other.null_mask[i]:
                return False

            if self.null_mask[i]:
                continue

            if memcmp(self.data + i * self.stride,
                      other.data + i * other.stride, self.stride) != 0:
                return False

        return True


cdef _box_pandas_bool(uint8_t* data, uint8_t* mask, int length,
                      copy=False):
    cdef:
        int i
        bint has_null = 0
        ndarray[object] oresult

    # Is there a null?
    with nogil:
        for i in range(length):
            if mask[i]:
                has_null = 1
                break

    if has_null:
        # Pack in object array
        oresult = np.empty(length, dtype=object)
        for i in range(length):
            if mask[i]:
                oresult[i] = None
            else:
                if data[i]:
                    oresult[i] = True
                else:
                    oresult[i] = False
        return oresult
    else:
        result = buffer_to_numpy_view(data, length, cnp.NPY_BOOL)
        if copy:
            result = result.copy()
        return result


cdef _box_pandas_integer(signed_int* data, uint8_t* mask, int length,
                         object dtype, copy=False):
    cdef:
        int i
        bint has_null = 0
        ndarray[double] fresult

    # Is there a null?
    with nogil:
        for i in range(length):
            if mask[i]:
                has_null = 1
                break

    if has_null:
        # Must pack in float64 array
        fresult = np.empty(length, dtype=np.float64)
        with nogil:
            for i in range(length):
                if mask[i]:
                    fresult[i] = NaN
                else:
                    # 64-bit integers could be outside the FP representation
                    # range, but that's always been an issue with pandas
                    fresult[i] = data[i]
        return fresult
    else:
        result = buffer_to_numpy_view(data, length, dtype.num)
        if copy:
            result = result.copy()
        return result

cdef _box_pandas_floating(cython.floating* data, uint8_t* mask, int length,
                          object dtype, copy=False):
    cdef:
        int i
        ndarray[cython.floating] result

    result = np.empty(length, dtype=dtype)
    with nogil:
        for i in range(length):
            if mask[i]:
                result[i] = NaN
            else:
                result[i] = data[i]
    return result


cdef _box_pandas_string(uint32_t* labels, uint8_t* mask, int length,
                        InternTable table):
    cdef:
        ndarray[object] result

    result = np.empty(length, dtype=object)

    return result


cdef buffer_to_numpy_view(void* data, int n, int ndtype):
    cdef:
        cnp.npy_intp shape[1]
        cnp.ndarray result

    # I believe this is a view by default
    shape[0] = <cnp.npy_intp> n
    result = cnp.PyArray_SimpleNewFromData(1, shape, ndtype, data)

    return result


def masked_from_numpy(ndarray values, ndarray mask, int ibis_type,
                      InternTableBuilder intern_t=None):
    # Helper function to convert masked format data represented as NumPy arrays
    # into a MaskedColumn which can be written out to an Ibis-format file
    cdef MaskedColumn result = MaskedColumn()

    check_numpy_compat(mask, IbisType.BOOLEAN)
    check_numpy_compat(values, ibis_type)

    if len(mask) != len(values):
        raise ValueError('arrays different lengths')

    # TODO: conversion of strings / other non-natively mapping types

    result.dtype = ibis_type
    result.stride = values.dtype.itemsize
    result.length = len(values)
    result.null_mask = <uint8_t*> mask.data
    result.data = <uint8_t*> values.data

    # Prevent these arrays from being garbage-collected
    result.obj_refs = [values, mask]

    return result


cdef class IbisTableWriter:
    """
    Writes the Ibis binary file format (in production this will be produced by
    Impala, but we need to be able to produce it ourselves mostly for testing
    purposes, and we can verify successful roundtrips to and from Impala
    separately.

    This class assumes that the passed columns are all formatted in the same
    way. If it becomes necessary, we can fairly easily revise the binary format
    to allow multiple formats per file.
    """
    cdef:
        object columns
        InternTable intern_table

        uint32_t ncols
        uint8_t* dtypes
        uint32_t* col_offsets
        uint32_t length
        uint8_t col_format

    def __cinit__(self, columns, InternTable intern_table=None):
        if len(columns) == 0:
            raise ValueError('must be at least one column')

        self.columns = columns
        self.intern_table = intern_table

        self.col_offsets = self.dtypes = NULL
        self._populate_metadata()

    def _populate_metadata(self):
        cdef:
            int i
            IbisColumn col

        self.length = len(self.columns[0])
        self.ncols = len(self.columns)

        self.dtypes = <uint8_t*> malloc(self.ncols)
        if self.dtypes == NULL:
            raise MemoryError

        self.col_offsets = <uint32_t*> malloc((self.ncols + 1) * 4)
        if self.col_offsets == NULL:
            free(self.dtypes)
            raise MemoryError

        cdef size_t offset = 0
        for i in range(self.ncols):
            col = self.columns[i]
            self.dtypes[i] = col.dtype
            self.col_offsets[i] = offset
            offset += col.nbytes()

        # "End cap" for easy arithmetic, also serves to mark the total number
        # of bytes for all columns
        self.col_offsets[self.ncols] = offset
        self.col_format = self.columns[0].format_code

    def total_size(self):
        cdef size_t total = 0

        # Total up preamble
        total = (
            4 +  # Magic
            4 +  # num columns
            8 +  # length
            self.ncols + # dtypes
            4 * (self.ncols + 1) + # column byte offsets
            1    # column format
        )

        # Add column bytes
        total += self.col_offsets[self.ncols]

        # TODO: Add intern table bytes

        return total

    def __dealloc__(self):
        if self.col_offsets != NULL:
            free(self.col_offsets)

        if self.dtypes != NULL:
            free(self.dtypes)

    def write(self, BufferLike obj):
        # Check there's enough space in the buffer object (e.g. a memory map)
        # to hold the results
        if obj.size < self.total_size():
            raise ValueError('Buffer is too small to hold the whole table')

        self.write_to(obj.get_buffer())

    cdef write_to(self, uint8_t* buf):
        cdef:
            BufferInterface writer = BufferInterface()
            IbisColumn col

        # Write table preamble
        writer.set_buffer(buf)

        writer.write_uint32(IMPALA_MAGIC_UINT32)
        writer.write_uint32(self.ncols)
        writer.write_uint64(self.length)

        writer.write_array(self.dtypes, self.ncols, 1)
        writer.write_array(self.col_offsets, self.ncols + 1, 4)
        writer.write_uint8(self.col_format)

        buf += writer.pos

        # Write columns
        for col in self.columns:
            col.write_buffer(buf)
            buf += col.nbytes()

        # Write string intern table and any other data
        if self.intern_table is not None:
            self.intern_table.write_buffer(buf)


cdef class BufferInterface:
    """
    File-like object for reading/writing bytes into some memory region
    """
    cdef:
        uint8_t* buf
        int pos

    def __cinit__(self):
        self.pos = 0

    cdef set_buffer(self, uint8_t* buf):
        self.buf = buf

    cdef seek(self, int pos):
        self.pos = pos

    cdef inline void write_array(self, void* data, int length, int stride):
        memcpy(self.buf + self.pos, data, length * stride)
        self.pos += length * stride

    cdef inline void write_uint8(self, uint8_t val):
        (self.buf + self.pos)[0] = val
        self.pos += 1

    cdef inline uint8_t read_uint8(self):
        cdef uint8_t val = (self.buf + self.pos)[0]
        self.pos += 1
        return val

    cdef inline void write_uint32(self, uint32_t val):
        (<uint32_t*> (self.buf + self.pos))[0] = val
        self.pos += 4

    cdef inline uint32_t read_uint32(self):
        cdef uint32_t val = (<uint32_t*> (self.buf + self.pos))[0]
        self.pos += 4
        return val

    cdef inline void write_uint64(self, uint64_t val):
        (<uint64_t*> (self.buf + self.pos))[0] = val
        self.pos += 8

    cdef inline uint64_t read_uint64(self):
        cdef uint64_t val = (<uint64_t*> (self.buf + self.pos))[0]
        self.pos += 8
        return val

#----------------------------------------------------------------------

# Faster UUIDs with libuuid

cdef extern from "Python.h":
    object _PyLong_FromByteArray(unsigned char *bytes, unsigned int n,
                                 int little_endian, int is_signed)
    char *PyString_AS_STRING(object s)


cdef class UUID:
    cdef:
        object bytes, int

    def __init__(self, version=4, *args, **kwargs):
        cdef object buf = cp.PyBytes_FromStringAndSize(NULL, 16)
        cdef unsigned char *_bytes = <unsigned char*>PyString_AS_STRING(buf)
        if version == 1:
            uuid_generate_time(_bytes)
        elif version == 4:
            uuid_generate_random(_bytes)
        self.bytes = buf
        self.int = _PyLong_FromByteArray(_bytes, 16, 0, 0)

    def get_bytes(self):
        return self.bytes

    def get_hex(self):
        return '%032x' % self.int

uuid = UUID

def uuid1_bytes():
    cdef object bytes = cp.PyBytes_FromStringAndSize(NULL, 16)
    uuid_generate_time(<unsigned char*>PyString_AS_STRING(bytes))
    return bytes

def uuid4_bytes():
    cdef object bytes = cp.PyBytes_FromStringAndSize(NULL, 16)
    uuid_generate_random(<unsigned char*>PyString_AS_STRING(bytes))
    return bytes

def uuid4_hex():
    cdef uuid_t guid
    uuid_generate_random(<unsigned char*> guid)
    return '%032x' % _PyLong_FromByteArray(<unsigned char*> guid, 16, 0, 0)
