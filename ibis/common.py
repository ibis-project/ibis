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


class IbisError(Exception):
    pass


class InternalError(IbisError):
    pass


class IntegrityError(IbisError):
    pass


class ExpressionError(IbisError):
    pass


class RelationError(ExpressionError):
    pass


class TranslationError(IbisError):
    pass


class OperationNotDefinedError(TranslationError):
    pass


class UnsupportedOperationError(TranslationError):
    pass


class UnsupportedBackendType(TranslationError):
    pass


class IbisInputError(ValueError, IbisError):
    pass


class IbisTypeError(TypeError, IbisError):
    pass


class InputTypeError(IbisTypeError):
    pass
