# Copyright 2015 Cloudera Inc
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from ibis.impala.udf import wrap_uda, wrap_udf
import ibis.expr.rules as rules


class MADLibAPI(object):

    """
    Class responsible for wrapping all MADLib-on-Impala API functions, creating
    them in a particular Impala database, and registering them for use with
    Ibis.
    """
    _udas = {
        'linr_fit': (['string', 'double'], 'string', 'LinrUpdate'),
        'logr_fit': (['string', 'string', 'boolean', 'double', 'double'],
                     'string', 'LogrUpdate'),
        'svm_fit': (['string', 'string', 'boolean', 'double', 'double'],
                    'string', 'SVMUpdate'),
    }

    _udfs = {
        'linr_predict': (['string', 'string'], 'double', 'LinrPredict'),

        'logr_predict': (['string', 'string'], 'boolean', 'LogrPredict'),
        'logr_loss': (['string', 'string', 'boolean'], 'double', 'LogrLoss'),

        'svm_predict': (['string', 'string'], 'boolean', 'SVMPredict'),
        'svm_loss': (['string', 'string', 'boolean'], 'double', 'SVMLoss'),

        'to_array': (rules.varargs(rules.double), 'string',
                     ('_Z7ToArrayPN10impala_udf'
                      '15FunctionContextEiPNS_9DoubleValE')),
        'arrayget': (['int64', 'string'], 'double', 'ArrayGet'),
        'allbytes': ([], 'string', 'AllBytes'),
        'printarray': (['string'], 'string', 'PrintArray'),
        'encodearray': (['string'], 'string', 'EncodeArray'),
        'decodearray': (['string'], 'string', 'DecodeArray'),
    }

    def __init__(self, library_path, database, func_prefix=None):
        self.library_path = library_path
        self.database = database

        self.function_names = sorted(self._udfs.keys() + self._udas.keys())
        self.func_prefix = func_prefix or 'madlib_'

        self._generate_wrappers()
        self._register_functions()

    def _generate_wrappers(self):
        for name, (inputs, output, update_sym) in self._udas.items():
            func = wrap_uda(self.library_path, inputs, output, update_sym,
                            name=self.func_prefix + name)
            setattr(self, name, func)

        for name, (inputs, output, sym) in self._udfs.items():
            func = wrap_udf(self.library_path, inputs, output, sym,
                            name=self.func_prefix + name)
            setattr(self, name, func)

    def _register_functions(self):
        # Enable SQL translation to work correctly
        for name in self.function_names:
            func = getattr(self, name)
            func.register(func.name, self.database)

    def create_functions(self, client):
        for name in self.function_names:
            func = getattr(self, name)
            client.create_function(func, database=self.database)

    def logistic_regression(self):
        pass

    def linear_regression(self):
        pass

    def svm(self):
        pass
