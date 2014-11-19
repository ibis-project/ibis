/*
   Copyright 2014 Cloudera Inc.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

#include <time.h>

/*
   Create an array of semaphores (e.g., 2) which can be used to coordinate IPC
   between two processes. Initializing with zeros indicates a locked state.

   @returns: semaphore id
 */
int semarray_init(int size, unsigned short* init_vals);

/* Returns 1 on success, 0 on failure */
int semarray_exists(int sem_id);

/*
  Delete semaphore set. Returns -1 on failure.
 */
int semarray_delete(int sem_id);

/*
  Try to perform a lock / unlock operation using the indicated timeout. If NULL
  is passed, the call will block until the operation is performed.

  TODO: semtimedop may not be supported on all systems,

  Returns -1 if the operation fails or times out (errno is EAGAIN in the latter
  case)
 */
int semarray_lock(int sem_id, int i, struct timespec *timeout);
int semarray_unlock(int sem_id, int i, struct timespec *timeout);
