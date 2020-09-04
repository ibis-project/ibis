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

import pytest

import ibis


@pytest.fixture
def pipe_table():
    return ibis.table(
        [
            ('key1', 'string'),
            ('key2', 'string'),
            ('key3', 'string'),
            ('value', 'double'),
        ],
        'foo_table',
    )


def test_pipe_positional_args(pipe_table):
    def my_func(data, foo, bar):
        return data[bar] + foo

    result = pipe_table.pipe(my_func, 4, 'value')
    expected = pipe_table['value'] + 4

    assert result.equals(expected)


def test_pipe_keyword_args(pipe_table):
    def my_func(data, foo=None, bar=None):
        return data[bar] + foo

    result = pipe_table.pipe(my_func, foo=4, bar='value')
    expected = pipe_table['value'] + 4

    assert result.equals(expected)


def test_pipe_pass_to_keyword(pipe_table):
    def my_func(x, y, data=None):
        return data[x] + y

    result = pipe_table.pipe((my_func, 'data'), 'value', 4)
    expected = pipe_table['value'] + 4

    assert result.equals(expected)


def test_call_pipe_equivalence(pipe_table):
    result = pipe_table(lambda x: x['key1'].cast('double').sum())
    expected = pipe_table.key1.cast('double').sum()
    assert result.equals(expected)
