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

from posix.time cimport timespec, time_t

cdef extern from "ipc_support.h" nogil:

    int semarray_init(int size, unsigned short* init_vals)
    int semarray_exists(int sem_id)
    int semarray_delete(int sem_id)

    int semarray_lock(int sem_id, int i, timespec* timeout)
    int semarray_unlock(int sem_id, int i, timespec* timeout)
