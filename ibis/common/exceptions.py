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
"""Module for exceptions."""

from __future__ import annotations

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


class BackendConversionError(IbisError):
    """A backend cannot convert an input to its native type."""


class BackendConfigurationNotRegistered(IbisError):
    """A backend has options but isn't registered in ibis/config.py."""

    def __init__(self, backend_name: str) -> None:
        super().__init__(backend_name)

    def __str__(self) -> str:
        (backend_name,) = self.args
        return f"Please register options for the `{backend_name}` backend in ibis/config.py"


class DuplicateUDFError(IbisError):
    def __init__(self, name: str) -> None:
        super().__init__(name)

    def __str__(self) -> str:
        (name,) = self.args
        return f"More than one function with `{name}` found."


class MissingUDFError(IbisError):
    def __init__(self, name: str) -> None:
        super().__init__(name)

    def __str__(self) -> str:
        (name,) = self.args
        return f"No user-defined function found with name `{name}`"


class AmbiguousUDFError(IbisError):
    def __init__(self, name: str) -> None:
        super().__init__(name)

    def __str__(self) -> str:
        (name,) = self.args
        return f"Multiple implementations of function `{name}`. Only one implementation is supported."


class MissingReturnAnnotationError(IbisError):
    def __init__(self, func_name: str):
        super().__init__(func_name)

    def __str__(self):
        (func_name,) = self.args
        return f"function `{func_name}` has no return type annotation"


class MissingParameterAnnotationError(IbisError):
    def __init__(self, func_name: str, param_name: str):
        super().__init__(func_name, param_name)

    def __str__(self):
        func_name, param_name = self.args
        return f"parameter `{param_name}` in function `{func_name}` is missing a type annotation"


class InvalidDecoratorError(IbisError):
    def __init__(self, name: str, line: str):
        super().__init__(name, line)

    def __str__(self) -> str:
        name, line = self.args
        return f"Only the `@udf` decorator is allowed in user-defined function: `{name}`; found line {line}"


def mark_as_unsupported(f: Callable) -> Callable:
    """Decorate an unsupported method."""

    # function that raises UnsupportedOperationError
    def _mark_as_unsupported(self):
        raise UnsupportedOperationError(
            f"Method `{f.__name__}` is unsupported by class `{self.__class__.__name__}`."
        )

    _mark_as_unsupported.__doc__ = f.__doc__
    _mark_as_unsupported.__name__ = f.__name__

    return _mark_as_unsupported
