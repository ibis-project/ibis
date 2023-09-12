from __future__ import annotations

import functools
import inspect
import types
from typing import TYPE_CHECKING, Callable
from typing import Any as AnyType

from ibis.common.bases import Immutable, Slotted
from ibis.common.patterns import (
    Any,
    FrozenDictOf,
    NoMatch,
    Option,
    Pattern,
    TupleOf,
)
from ibis.common.patterns import pattern as ensure_pattern
from ibis.common.typing import format_typehint, get_type_hints

if TYPE_CHECKING:
    from collections.abc import Sequence

EMPTY = inspect.Parameter.empty  # marker for missing argument
KEYWORD_ONLY = inspect.Parameter.KEYWORD_ONLY
POSITIONAL_ONLY = inspect.Parameter.POSITIONAL_ONLY
POSITIONAL_OR_KEYWORD = inspect.Parameter.POSITIONAL_OR_KEYWORD
VAR_KEYWORD = inspect.Parameter.VAR_KEYWORD
VAR_POSITIONAL = inspect.Parameter.VAR_POSITIONAL


_any = Any()


class ValidationError(Exception):
    __slots__ = ()


class AttributeValidationError(ValidationError):
    __slots__ = ("name", "value", "pattern")

    def __init__(self, name: str, value: AnyType, pattern: Pattern):
        self.name = name
        self.value = value
        self.pattern = pattern

    def __str__(self):
        return f"Failed to validate attribute `{self.name}`: {self.value!r} is not {self.pattern.describe()}"


class ReturnValidationError(ValidationError):
    __slots__ = ("func", "value", "pattern")

    def __init__(self, func: Callable, value: AnyType, pattern: Pattern):
        self.func = func
        self.value = value
        self.pattern = pattern

    def __str__(self):
        return f"Failed to validate return value of `{self.func.__name__}`: {self.value!r} is not {self.pattern.describe()}"


class SignatureValidationError(ValidationError):
    __slots__ = ("msg", "sig", "func", "args", "kwargs", "errors")

    def __init__(
        self,
        msg: str,
        sig: Signature,
        func: Callable,
        args: tuple[AnyType, ...],
        kwargs: dict[str, AnyType],
        errors: Sequence[tuple[str, AnyType, Pattern]] = (),
    ):
        self.msg = msg
        self.sig = sig
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.errors = errors

    def __str__(self):
        args = tuple(repr(arg) for arg in self.args)
        args += tuple(f"{k}={v!r}" for k, v in self.kwargs.items())
        call = f"{self.func.__name__}({', '.join(args)})"

        errors = ""
        for name, value, pattern in self.errors:
            errors += f"\n  `{name}`: {value!r} is not {pattern.describe()}"

        sig = f"{self.func.__name__}{self.sig}"
        cause = str(self.__cause__) if self.__cause__ else ""

        return self.msg.format(sig=sig, call=call, cause=cause, errors=errors)


class Annotation(Slotted, Immutable):
    """Base class for all annotations.

    Annotations are used to mark fields in a class and to validate them.
    """

    __slots__ = ()

    def validate(self, name: str, value: AnyType, this: AnyType) -> AnyType:
        """Validate the field.

        Parameters
        ----------
        name
            The name of the attribute.
        value
            The value of the attribute.
        this
            The instance of the class the attribute is defined on.

        Returns
        -------
        The validated value for the field.
        """
        result = self.pattern.match(value, this)
        if result is NoMatch:
            raise AttributeValidationError(
                name=name,
                value=value,
                pattern=self.pattern,
            )
        return result


class Attribute(Annotation):
    """Annotation to mark a field in a class.

    An optional pattern can be provider to validate the field every time it
    is set.

    Parameters
    ----------
    pattern : Pattern, default noop
        Pattern to validate the field.
    default : Callable, default EMPTY
        Callable to compute the default value of the field.
    """

    __slots__ = ("pattern", "default")
    pattern: Pattern
    default: AnyType

    def __init__(self, pattern: Pattern = _any, default: AnyType = EMPTY):
        super().__init__(pattern=ensure_pattern(pattern), default=default)

    def has_default(self):
        """Check if the field has a default value.

        Returns
        -------
        bool
        """
        return self.default is not EMPTY

    def get_default(self, name: str, this: AnyType) -> AnyType:
        """Get the default value of the field.

        Parameters
        ----------
        name
            The name of the attribute.
        this
            The instance of the class the attribute is defined on.

        Returns
        -------
        The default value for the field.
        """
        if callable(self.default):
            value = self.default(this)
        else:
            value = self.default
        return self.validate(name, value, this)

    def __call__(self, default):
        """Needed to support the decorator syntax."""
        return self.__class__(self.pattern, default)


class Argument(Annotation):
    """Annotation type for all fields which should be passed as arguments.

    Parameters
    ----------
    pattern
        Optional pattern to validate the argument.
    default
        Optional default value of the argument.
    typehint
        Optional typehint of the argument.
    kind
        Kind of the argument, one of `inspect.Parameter` constants.
        Defaults to positional or keyword.
    """

    __slots__ = ("pattern", "default", "typehint", "kind")
    pattern: Pattern
    default: AnyType
    typehint: AnyType
    kind: int

    def __init__(
        self,
        pattern: Pattern = _any,
        default: AnyType = EMPTY,
        typehint: type | None = None,
        kind: int = POSITIONAL_OR_KEYWORD,
    ):
        super().__init__(
            pattern=ensure_pattern(pattern),
            default=default,
            typehint=typehint,
            kind=kind,
        )


def attribute(pattern=_any, default=EMPTY):
    """Annotation to mark a field in a class."""
    if default is EMPTY and isinstance(pattern, (types.FunctionType, types.MethodType)):
        return Attribute(default=pattern)
    else:
        return Attribute(pattern, default=default)


def argument(pattern=_any, default=EMPTY, typehint=None):
    """Annotation type for all fields which should be passed as arguments."""
    return Argument(pattern, default=default, typehint=typehint)


def optional(pattern=_any, default=None, typehint=None):
    """Annotation to allow and treat `None` values as missing arguments."""
    if pattern is None:
        pattern = Option(Any(), default=default)
    else:
        pattern = Option(pattern, default=default)
    return Argument(pattern, default=None, typehint=typehint)


def varargs(pattern=_any, typehint=None):
    """Annotation to mark a variable length positional arguments."""
    return Argument(TupleOf(pattern), kind=VAR_POSITIONAL, typehint=typehint)


def varkwargs(pattern=_any, typehint=None):
    """Annotation to mark a variable length keyword arguments."""
    return Argument(FrozenDictOf(_any, pattern), kind=VAR_KEYWORD, typehint=typehint)


class Parameter(inspect.Parameter):
    """Augmented Parameter class to additionally hold a pattern object."""

    __slots__ = ()

    def __init__(self, name, annotation):
        if not isinstance(annotation, Argument):
            raise TypeError(
                f"annotation must be an instance of Argument, got {annotation}"
            )
        super().__init__(
            name,
            kind=annotation.kind,
            default=annotation.default,
            annotation=annotation,
        )

    def __str__(self):
        formatted = self._name

        if self._annotation is not EMPTY:
            typehint = format_typehint(self._annotation.typehint)
            formatted = f"{formatted}: {typehint}"

        if self._default is not EMPTY:
            if self._annotation is not EMPTY:
                formatted = f"{formatted} = {self._default!r}"
            else:
                formatted = f"{formatted}={self._default!r}"

        if self._kind == VAR_POSITIONAL:
            formatted = "*" + formatted
        elif self._kind == VAR_KEYWORD:
            formatted = "**" + formatted

        return formatted


class Signature(inspect.Signature):
    """Validatable signature.

    Primarily used in the implementation of `ibis.common.grounds.Annotable`.
    """

    __slots__ = ()

    @classmethod
    def merge(cls, *signatures, **annotations):
        """Merge multiple signatures.

        In addition to concatenating the parameters, it also reorders the
        parameters so that optional arguments come after mandatory arguments.

        Parameters
        ----------
        *signatures : Signature
            Signature instances to merge.
        **annotations : dict
            Annotations to add to the merged signature.

        Returns
        -------
        Signature
        """
        params = {}
        for sig in signatures:
            params.update(sig.parameters)

        inherited = set(params.keys())
        for name, annot in annotations.items():
            params[name] = Parameter(name, annotation=annot)

        # mandatory fields without default values must precede the optional
        # ones in the function signature, the partial ordering will be kept
        var_args, var_kwargs = [], []
        new_args, new_kwargs = [], []
        old_args, old_kwargs = [], []

        for name, param in params.items():
            if param.kind == VAR_POSITIONAL:
                if var_args:
                    raise TypeError("only one variadic *args parameter is allowed")
                var_args.append(param)
            elif param.kind == VAR_KEYWORD:
                if var_kwargs:
                    raise TypeError("only one variadic **kwargs parameter is allowed")
                var_kwargs.append(param)
            elif name in inherited:
                if param.default is EMPTY:
                    old_args.append(param)
                else:
                    old_kwargs.append(param)
            elif param.default is EMPTY:
                new_args.append(param)
            else:
                new_kwargs.append(param)

        return cls(
            old_args + new_args + var_args + new_kwargs + old_kwargs + var_kwargs
        )

    @classmethod
    def from_callable(cls, fn, patterns=None, return_pattern=None):
        """Create a validateable signature from a callable.

        Parameters
        ----------
        fn : Callable
            Callable to create a signature from.
        patterns : list or dict, default None
            Pass patterns to add missing or override existing argument type
            annotations.
        return_pattern : Pattern, default None
            Pattern for the return value of the callable.

        Returns
        -------
        Signature
        """
        sig = super().from_callable(fn)
        typehints = get_type_hints(fn)

        if patterns is None:
            patterns = {}
        elif isinstance(patterns, (list, tuple)):
            # create a mapping of parameter name to pattern
            patterns = dict(zip(sig.parameters.keys(), patterns))
        elif not isinstance(patterns, dict):
            raise TypeError(f"patterns must be a list or dict, got {type(patterns)}")

        parameters = []
        for param in sig.parameters.values():
            name = param.name
            kind = param.kind
            default = param.default
            typehint = typehints.get(name)

            if name in patterns:
                pattern = patterns[name]
            elif typehint is not None:
                pattern = Pattern.from_typehint(typehint)
            else:
                pattern = _any

            if kind is VAR_POSITIONAL:
                annot = varargs(pattern, typehint=typehint)
            elif kind is VAR_KEYWORD:
                annot = varkwargs(pattern, typehint=typehint)
            else:
                annot = Argument(pattern, kind=kind, default=default, typehint=typehint)

            parameters.append(Parameter(param.name, annot))

        if return_pattern is not None:
            return_annotation = return_pattern
        elif (typehint := typehints.get("return")) is not None:
            return_annotation = Pattern.from_typehint(typehint)
        else:
            return_annotation = EMPTY

        return cls(parameters, return_annotation=return_annotation)

    def unbind(self, this: dict[str, Any]) -> tuple[tuple[Any, ...], dict[str, Any]]:
        """Reverse bind of the parameters.

        Attempts to reconstructs the original arguments as keyword only arguments.

        Parameters
        ----------
        this : Any
            Object with attributes matching the signature parameters.

        Returns
        -------
        args : (args, kwargs)
            Tuple of positional and keyword arguments.
        """
        # does the reverse of bind, but doesn't apply defaults
        args: list = []
        kwargs: dict = {}
        for name, param in self.parameters.items():
            value = this[name]
            if param.kind is POSITIONAL_OR_KEYWORD:
                args.append(value)
            elif param.kind is VAR_POSITIONAL:
                args.extend(value)
            elif param.kind is VAR_KEYWORD:
                kwargs.update(value)
            elif param.kind is KEYWORD_ONLY:
                kwargs[name] = value
            elif param.kind is POSITIONAL_ONLY:
                args.append(value)
            else:
                raise TypeError(f"unsupported parameter kind {param.kind}")
        return tuple(args), kwargs

    def validate(self, func, args, kwargs):
        """Validate the arguments against the signature.

        Parameters
        ----------
        func : Callable
            Callable to validate the arguments for.
        args : tuple
            Positional arguments.
        kwargs : dict
            Keyword arguments.

        Returns
        -------
        validated : dict
            Dictionary of validated arguments.
        """
        try:
            bound = self.bind(*args, **kwargs)
            bound.apply_defaults()
        except TypeError as err:
            raise SignatureValidationError(
                "{call} {cause}\n\nExpected signature: {sig}",
                sig=self,
                func=func,
                args=args,
                kwargs=kwargs,
            ) from err

        this, errors = {}, []
        for name, value in bound.arguments.items():
            param = self.parameters[name]
            pattern = param.annotation.pattern

            result = pattern.match(value, this)
            if result is NoMatch:
                errors.append((name, value, pattern))
            else:
                this[name] = result

        if errors:
            raise SignatureValidationError(
                "{call} has failed due to the following errors:{errors}\n\nExpected signature: {sig}",
                sig=self,
                func=func,
                args=args,
                kwargs=kwargs,
                errors=errors,
            )

        return this

    def validate_nobind(self, func, kwargs):
        """Validate the arguments against the signature without binding."""
        this, errors = {}, []
        for name, param in self.parameters.items():
            value = kwargs.get(name, param.default)
            if value is EMPTY:
                raise TypeError(f"missing required argument `{name!r}`")

            pattern = param.annotation.pattern
            result = pattern.match(value, this)
            if result is NoMatch:
                errors.append((name, value, pattern))
            else:
                this[name] = result

        if errors:
            raise SignatureValidationError(
                "{call} has failed due to the following errors:{errors}\n\nExpected signature: {sig}",
                sig=self,
                func=func,
                args=(),
                kwargs=kwargs,
                errors=errors,
            )

        return this

    def validate_return(self, func, value):
        """Validate the return value of a function.

        Parameters
        ----------
        func : Callable
            Callable to validate the return value for.
        value : Any
            Return value of the function.

        Returns
        -------
        validated : Any
            Validated return value.
        """
        if self.return_annotation is EMPTY:
            return value

        result = self.return_annotation.match(value, {})
        if result is NoMatch:
            raise ReturnValidationError(
                func=func,
                value=value,
                pattern=self.return_annotation,
            )

        return result


def annotated(_1=None, _2=None, _3=None, **kwargs):
    """Create functions with arguments validated at runtime.

    There are various ways to apply this decorator:

    1. With type annotations

    >>> @annotated
    ... def foo(x: int, y: str) -> float:
    ...     return float(x) + float(y)

    2. With argument patterns passed as keyword arguments

    >>> from ibis.common.patterns import InstanceOf as instance_of
    >>> @annotated(x=instance_of(int), y=instance_of(str))
    ... def foo(x, y):
    ...     return float(x) + float(y)
    ...

    3. With mixing type annotations and patterns where the latter takes precedence

    >>> @annotated(x=instance_of(float))
    ... def foo(x: int, y: str) -> float:
    ...     return float(x) + float(y)

    4. With argument patterns passed as a list and/or an optional return pattern

    >>> @annotated([instance_of(int), instance_of(str)], instance_of(float))
    ... def foo(x, y):
    ...     return float(x) + float(y)
    ...

    Parameters
    ----------
    *args : Union[
                tuple[Callable],
                tuple[list[Pattern], Callable],
                tuple[list[Pattern], Pattern, Callable]
            ]
        Positional arguments.
        - If a single callable is passed, it's wrapped with the signature
        - If two arguments are passed, the first one is a list of patterns for the
          arguments and the second one is the callable to wrap
        - If three arguments are passed, the first one is a list of patterns for the
          arguments, the second one is a pattern for the return value and the third
          one is the callable to wrap
    **kwargs : dict[str, Pattern]
        Patterns for the arguments.

    Returns
    -------
    Callable
    """
    if _1 is None:
        return functools.partial(annotated, **kwargs)
    elif _2 is None:
        if callable(_1):
            func, patterns, return_pattern = _1, None, None
        else:
            return functools.partial(annotated, _1, **kwargs)
    elif _3 is None:
        if not isinstance(_2, Pattern):
            func, patterns, return_pattern = _2, _1, None
        else:
            return functools.partial(annotated, _1, _2, **kwargs)
    else:
        func, patterns, return_pattern = _3, _1, _2

    sig = Signature.from_callable(
        func, patterns=patterns or kwargs, return_pattern=return_pattern
    )

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        # 1. Validate the passed arguments
        values = sig.validate(func, args, kwargs)
        # 2. Reconstruction of the original arguments
        args, kwargs = sig.unbind(values)
        # 3. Call the function with the validated arguments
        result = func(*args, **kwargs)
        # 4. Validate the return value
        return sig.validate_return(func, result)

    wrapped.__signature__ = sig

    return wrapped
