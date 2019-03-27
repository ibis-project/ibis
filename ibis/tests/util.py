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

import pytest

import ibis
import ibis.common as com
import ibis.util as util

from ibis.compat import functools


def assert_equal(left, right):
    if util.all_of([left, right], ibis.Schema):
        assert left.equals(right),\
            'Comparing schemas: \n%s !=\n%s' % (repr(left), repr(right))
    else:
        assert left.equals(right), ('Objects unequal: {0}\nvs\n{1}'
                                    .format(repr(left), repr(right)))


def skipif_unsupported(f):
    @functools.wraps(f)
    def wrapper(backend, *args, **kwargs):
        try:
            return f(backend, *args, **kwargs)
        except (com.OperationNotDefinedError,
                com.UnsupportedOperationError,
                com.UnsupportedBackendType,
                NotImplementedError) as e:
            pytest.skip('{} using {}'.format(e, str(backend)))
    return wrapper


def skipif_backend(skip_backend):
    def wrapped(f):
        @functools.wraps(f)
        def wrapper(backend, *args, **kwargs):
            if isinstance(backend, skip_backend):
                pytest.skip('Skipping {} test'.format(str(backend)))
            else:
                return f(backend, *args, **kwargs)
        return wrapper
    return wrapped
