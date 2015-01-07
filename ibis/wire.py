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

import struct

try:
    from cStringIO import StringIO as BytesIO
except ImportError:
    from io import BytesIO


class PackedMessageReader(object):

    """

    """

    def __init__(self, msg):
        self.msg = msg

        if isinstance(msg, basestring):
            self.buf = BytesIO(msg)
        elif hasattr(msg, 'read'):
            self.buf = msg
        else:
            raise TypeError('Cannot read from input type: %s' % type(msg))

    def string(self):
        """
        uint32_t prefixed
        """
        return read_string(self.buf)

    def uint8(self):
        return self._unpack('b', 1)

    def uint32(self):
        return self._unpack('I', 4)

    def uint64(self):
        return self._unpack('Q', 8)

    def _unpack(self, fmt, size):
        return struct.unpack(fmt, self.buf.read(size))[0]


class PackedMessageWriter(object):

    """

    """

    def __init__(self, buf=None):
        if buf is None:
            buf = BytesIO()

        self.buf = buf

    def get_result(self):
        return self.buf.getvalue()

    def string(self, val):
        write_string(self.buf, val)

    def uint8(self, val):
        write_uint8(self.buf, val)
        return self

    def uint32(self, val):
        write_uint32(self.buf, val)
        return self

    def uint64(self, val):
        write_uint64(self.buf, val)
        return self


def write_string(buf, val):
    write_uint32(buf, len(val))
    buf.write(val)


def read_string(buf):
    slen = _read(buf, 'I', 4)
    return buf.read(slen)


def write_uint8(buf, val):
    _write(buf, 'b', val)


def write_uint32(buf, val):
    _write(buf, 'I', val)


def write_uint64(buf, val):
    _write(buf, 'Q', val)


def _write(buf, fmt, val):
    buf.write(struct.pack(fmt, val))


def _read(buf, fmt, size):
    return struct.unpack(fmt, buf.read(size))[0]
