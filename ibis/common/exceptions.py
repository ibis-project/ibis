"""Module for exceptions."""
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
from typing import Callable


class IbisError(Exception):
    """IbisError."""


class InternalError(IbisError):
    """InternalError."""


class IntegrityError(IbisError):
    """IntegrityError."""


class ExpressionError(IbisError):
    """ExpressionError."""


class RelationError(ExpressionError):
    """RelationError."""


class TranslationError(IbisError):
    """TranslationError."""


class OperationNotDefinedError(TranslationError):
    """OperationNotDefinedError."""


class UnsupportedOperationError(TranslationError):
    """UnsupportedOperationError."""


class UnsupportedBackendType(TranslationError):
    """UnsupportedBackendType."""


class UnboundExpressionError(ValueError, IbisError):
    """UnboundExpressionError."""


class IbisInputError(ValueError, IbisError):
    """IbisInputError."""


class IbisTypeError(TypeError, IbisError):
    """IbisTypeError."""


class InputTypeError(IbisTypeError):
    """InputTypeError."""


class UnsupportedArgumentError(IbisError):
    """UnsupportedArgumentError."""


def mark_as_unsupported(f: Callable) -> Callable:
    """Decorate an unsupported method.

    Parameters
    ----------
    f : callable

    Returns
    -------
    callable

    Raises
    ------
    UnsupportedOperationError
    """
    # function that raises UnsupportedOperationError
    def _mark_as_unsupported(self):
        raise UnsupportedOperationError(
            'Method `{}` are unsupported by class `{}`.'.format(
                f.__name__, self.__class__.__name__
            )
        )

    _mark_as_unsupported.__doc__ = f.__doc__
    _mark_as_unsupported.__name__ = f.__name__

    return _mark_as_unsupported
