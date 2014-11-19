from ibis.comms import IPCLock
import threading

class Counter:

    def __init__(self):
        self.total = 0

    def inc(self):
        self.total += 1

def run_test():
    counter = Counter()
    ceiling = 1000

    master = IPCLock(is_slave=0)
    slave = IPCLock(master.semaphore_id)

    def ping():
        while True:
            with slave:
                counter.inc()
                if counter.total > ceiling:
                    break

    def pong():
        while True:
            with master:
                counter.inc()
                if counter.total > ceiling:
                    break

    t1 = threading.Thread(target=pong)
    t1.start()

    t2 = threading.Thread(target=ping)
    t2.start()

    t1.join()
    t2.join()
