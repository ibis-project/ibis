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

#include "Python.h"

#include "ipc_support.h"
#include <errno.h>
#include <sys/shm.h>
#include <sys/sem.h>

/* This union is not defined in sys/sem.h on some systems */
#if (!__APPLE__) && (!defined(__GNU_LIBRARY__) || defined(_SEM_SEMUN_UNDEFINED))
union semun {
  int val;               /* value for SETVAL */
  struct semid_ds* buf;  /* buffer for IPC_STAT, IPC_SET */
  unsigned short* array; /* array for GETALL, SETALL */
  struct seminfo* __buf; /* buffer for IPC_INFO; linux only */
};
#endif

#if __APPLE__
int semtimedop(int semid, struct sembuf *sops, unsigned nsops,
               struct timespec *timeout) {
  return semop(semid, sops, nsops);
}
#endif

int semarray_init(int size, unsigned short* init_vals) {
  union semun arg;
  int sem_id;

  arg.array = init_vals;
  sem_id = semget(IPC_PRIVATE, size, SHM_W | SHM_R);
  semctl(sem_id, 0, SETALL, arg);
  return sem_id;
}

int semarray_exists(int sem_id) {
  union semun arg;
  struct semid_ds statinfo;
  int ret;

  /* Is this the best way? */
  arg.buf = &statinfo;
  ret = semctl(sem_id, 0, IPC_STAT, arg);
  return ret >= 0 ? 1 : 0;
}

int semarray_delete(int sem_id) {
  return semctl(sem_id, 0, IPC_RMID, NULL);
}

int semarray_lock(int sem_id, int i, struct timespec *timeout) {
  struct sembuf buf;

  /* Decrement is locking */
  buf.sem_op = -1;
  buf.sem_num = i;
  buf.sem_flg = SEM_UNDO;
  return semtimedop(sem_id, &buf, 1, timeout);
}

int semarray_unlock(int sem_id, int i, struct timespec *timeout) {
  struct sembuf buf;

  /* Increment is unlocking */
  buf.sem_op = 1;
  buf.sem_num = i;
  buf.sem_flg = SEM_UNDO;
  return semtimedop(sem_id, &buf, 1, timeout);
}
