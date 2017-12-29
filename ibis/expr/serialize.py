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

import ibis
from ibis.compat import pickle


def serialize(expr, metadata=None):
    """
    serialize the ibis expression to a textual form

    Parameters
    ----------
    expr : expression
    metadata : python expression, optional

    Returns
    -------
    byte string of serialized expression
    """

    d = {'version': ibis.__version__,
         'metadata': metadata,
         'format': 'pickle',
         'expr': expr}
    return pickle.dumps(d)


def deserialize(s):
    """
    de-serialize an ibis expression, returning a
    dictionary of the expression and metadata

    Parameters
    ----------
    s : byte string

    Returns
    -------
    dict of expression & metadata
    """

    return pickle.loads(s)
