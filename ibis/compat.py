# Copyright 2015 Cloudera Inc.
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

# flake8: noqa

import sys
from six import BytesIO, StringIO, string_types as py_string


PY26 = sys.version_info[0] == 2 and sys.version_info[1] == 6
PY3 = (sys.version_info[0] >= 3)
PY2 = sys.version_info[0] == 2


if PY26:
    import unittest2 as unittest
else:
    import unittest

if PY3:
    unicode_type = str
else:
    unicode_type = unicode
