#!/usr/bin/env python

import json
import socket
import sys

import asv

if __name__ == '__main__':
    if len(sys.argv) > 1:
        hostname = sys.argv[1]
    else:
        hostname = socket.gethostname()

    machine_info = asv.machine.Machine.get_defaults()
    machine_info['machine'] = hostname
    machine_info['ram'] = '{:d}GB'.format(int(machine_info['ram']) // 1000000)
    print(json.dumps({hostname: machine_info, 'version': 1}, indent=2))
