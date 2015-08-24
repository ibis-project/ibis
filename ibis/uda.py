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

import base64


class UDAEvaluator(object):

    def __init__(self, uda_class, view_name, uda_fields, key_fields=None):
        from ibis.compat import pickle_dump
        self.uda_class = uda_class

        self.uda_encoded = base64.b64_encode(pickle_dump(uda_class))

        self.view_name = view_name
        self.uda_fields = uda_fields

        self.key_fields = key_fields

    def get_result(self, cursor):
        pass


def uda_evaluate(cursor, uda_class, view_name, uda_fields, key_fields=None):
    evaluator = UDAEvaluator(uda_class, view_name, uda_fields,
                             key_fields=key_fields)
    return evaluator.get_result(cursor)
