from __future__ import annotations

import functools
import inspect
from typing import Any as AnyType

from ibis.common.collections import DotDict
from ibis.common.patterns import (
    Any,
    FrozenDictOf,
    Function,
    Option,
    TupleOf,
    Validator,
)
from ibis.common.typing import get_type_hints

EMPTY = inspect.Parameter.empty  # marker for missing argument
KEYWORD_ONLY = inspect.Parameter.KEYWORD_ONLY
POSITIONAL_ONLY = inspect.Parameter.POSITIONAL_ONLY
POSITIONAL_OR_KEYWORD = inspect.Parameter.POSITIONAL_OR_KEYWORD
VAR_KEYWORD = inspect.Parameter.VAR_KEYWORD
VAR_POSITIONAL = inspect.Parameter.VAR_POSITIONAL


class Annotation:
    """Base class for all annotations.

    Annotations are used to mark fields in a class and to validate them.

    Parameters
    ----------
    validator : Validator, default noop
        Validator to validate the field.
    default : Any, default EMPTY
        Default value of the field.
    typehint : type, default EMPTY
        Type of the field, not used for validation.
    """

    __slots__ = ('_validator', '_default', '_typehint')

    def __init__(self, validator=None, default=EMPTY, typehint=EMPTY):
        if validator is None or isinstance(validator, Validator):
            pass
        elif callable(validator):
            validator = Function(validator)
        else:
            raise TypeError(f"Unsupported validator {validator!r}")
        self._default = default
        self._typehint = typehint
        self._validator = validator

    def __eq__(self, other):
        return (
            type(self) is type(other)
            and self._default == other._default
            and self._typehint == other._typehint
            and self._validator == other._validator
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(validator={self._validator!r}, "
            f"default={self._default!r}, typehint={self._typehint!r})"
        )

    def validate(self, arg, context=None):
        if self._validator is None:
            return arg
        return self._validator.validate(arg, context)


class Attribute(Annotation):
    """Annotation to mark a field in a class.

    An optional validator can be provider to validate the field every time it
    is set.

    Parameters
    ----------
    validator : Validator, default noop
        Validator to validate the field.
    default : Callable, default EMPTY
        Callable to compute the default value of the field.
    """

    @classmethod
    def default(self, fn):
        """Annotation to mark a field with a default value computed by a callable."""
        return Attribute(default=fn)

    def initialize(self, this):
        """Compute the default value of the field."""
        if self._default is EMPTY:
            return EMPTY
        elif callable(self._default):
            value = self._default(this)
        else:
            value = self._default
        return self.validate(value, this)


class Argument(Annotation):
    """Annotation type for all fields which should be passed as arguments.

    Parameters
    ----------
    validator
        Optional validator to validate the argument.
    default
        Optional default value of the argument.
    typehint
        Optional typehint of the argument.
    kind
        Kind of the argument, one of `inspect.Parameter` constants.
        Defaults to positional or keyword.
    """

    __slots__ = ('_kind',)

    def __init__(
        self,
        validator: Validator | None = None,
        default: AnyType = EMPTY,
        typehint: type | None = None,
        kind: int = POSITIONAL_OR_KEYWORD,
    ):
        super().__init__(validator, default, typehint)
        self._kind = kind

    @classmethod
    def required(cls, validator=None, **kwargs):
        """Annotation to mark a mandatory argument."""
        return cls(validator, **kwargs)

    @classmethod
    def default(cls, default, validator=None, **kwargs):
        """Annotation to allow missing arguments with a default value."""
        return cls(validator, default, **kwargs)

    @classmethod
    def optional(cls, validator=None, default=None, **kwargs):
        """Annotation to allow and treat `None` values as missing arguments."""
        if validator is None:
            validator = Option(Any(), default=default)
        else:
            validator = Option(validator, default=default)
        return cls(validator, default=None, **kwargs)

    @classmethod
    def varargs(cls, validator=None, **kwargs):
        """Annotation to mark a variable length positional argument."""
        validator = None if validator is None else TupleOf(validator)
        return cls(validator, kind=VAR_POSITIONAL, **kwargs)

    @classmethod
    def varkwargs(cls, validator=None, **kwargs):
        validator = None if validator is None else FrozenDictOf(Any(), validator)
        return cls(validator, kind=VAR_KEYWORD, **kwargs)


class Parameter(inspect.Parameter):
    """Augmented Parameter class to additionally hold a validator object."""

    __slots__ = ()

    def __init__(self, name, annotation):
        if not isinstance(annotation, Argument):
            raise TypeError(
                f'annotation must be an instance of Argument, got {annotation}'
            )
        super().__init__(
            name,
            kind=annotation._kind,
            default=annotation._default,
            annotation=annotation,
        )


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
                    raise TypeError('only one variadic *args parameter is allowed')
                var_args.append(param)
            elif param.kind == VAR_KEYWORD:
                if var_kwargs:
                    raise TypeError('only one variadic **kwargs parameter is allowed')
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
    def from_callable(cls, fn, validators=None, return_validator=None):
        """Create a validateable signature from a callable.

        Parameters
        ----------
        fn : Callable
            Callable to create a signature from.
        validators : list or dict, default None
            Pass validators to add missing or override existing argument type
            annotations.
        return_validator : Validator, default None
            Validator for the return value of the callable.

        Returns
        -------
        Signature
        """
        sig = super().from_callable(fn)
        typehints = get_type_hints(fn)

        if validators is None:
            validators = {}
        elif isinstance(validators, (list, tuple)):
            # create a mapping of parameter name to validator
            validators = dict(zip(sig.parameters.keys(), validators))
        elif not isinstance(validators, dict):
            raise TypeError(
                f'validators must be a list or dict, got {type(validators)}'
            )

        parameters = []
        for param in sig.parameters.values():
            name = param.name
            kind = param.kind
            default = param.default
            typehint = typehints.get(name)

            if name in validators:
                validator = validators[name]
            elif typehint is not None:
                validator = Validator.from_typehint(typehint)
            else:
                validator = None

            if kind is VAR_POSITIONAL:
                annot = Argument.varargs(validator, typehint=typehint)
            elif kind is VAR_KEYWORD:
                annot = Argument.varkwargs(validator, typehint=typehint)
            elif default is EMPTY:
                annot = Argument.required(validator, kind=kind, typehint=typehint)
            else:
                annot = Argument.default(
                    default, validator, kind=param.kind, typehint=typehint
                )

            parameters.append(Parameter(param.name, annot))

        if return_validator is not None:
            return_annotation = return_validator
        elif (typehint := typehints.get("return")) is not None:
            return_annotation = Validator.from_typehint(typehint)
        else:
            return_annotation = EMPTY

        return cls(parameters, return_annotation=return_annotation)

    def unbind(self, this: AnyType):
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
        args, kwargs = [], {}
        for name, param in self.parameters.items():
            value = getattr(this, name)
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

    def validate(self, *args, **kwargs):
        """Validate the arguments against the signature.

        Parameters
        ----------
        args : tuple
            Positional arguments.
        kwargs : dict
            Keyword arguments.

        Returns
        -------
        validated : dict
            Dictionary of validated arguments.
        """
        # bind the signature to the passed arguments and apply the validators
        # before passing the arguments, so self.__init__() receives already
        # validated arguments as keywords
        bound = self.bind(*args, **kwargs)
        bound.apply_defaults()

        this = DotDict()
        for name, value in bound.arguments.items():
            param = self.parameters[name]
            # TODO(kszucs): provide more error context on failure
            this[name] = param.annotation.validate(value, this)

        return this

    def validate_nobind(self, **kwargs):
        """Validate the arguments against the signature without binding."""
        this = DotDict()
        for name, param in self.parameters.items():
            value = kwargs.get(name, param.default)
            if value is EMPTY:
                raise TypeError(f"missing required argument `{name!r}`")
            this[name] = param.annotation.validate(value, kwargs)
        return this

    def validate_return(self, value, context):
        """Validate the return value of a function.

        Parameters
        ----------
        value : Any
            Return value of the function.
        context : dict
            Context dictionary.

        Returns
        -------
        validated : Any
            Validated return value.
        """
        if self.return_annotation is EMPTY:
            return value
        return self.return_annotation.validate(value, context)


# aliases for convenience
argument = Argument
attribute = Attribute
default = Argument.default
optional = Argument.optional
required = Argument.required
varargs = Argument.varargs
varkwargs = Argument.varkwargs


# TODO(kszucs): try to cache validator objects
# TODO(kszucs): try a quicker curry implementation


def annotated(_1=None, _2=None, _3=None, **kwargs):
    """Create functions with arguments validated at runtime.

    There are various ways to apply this decorator:

    1. With type annotations

    >>> @annotated
    ... def foo(x: int, y: str) -> float:
    ...     return float(x) + float(y)

    2. With argument validators passed as keyword arguments

    >>> from ibis.common.patterns import InstanceOf as instance_of
    >>> @annotated(x=instance_of(int), y=instance_of(str))
    ... def foo(x, y):
    ...     return float(x) + float(y)

    3. With mixing type annotations and validators where the latter takes precedence

    >>> @annotated(x=instance_of(float))
    ... def foo(x: int, y: str) -> float:
    ...     return float(x) + float(y)

    4. With argument validators passed as a list and/or an optional return validator

    >>> @annotated([instance_of(int), instance_of(str)], instance_of(float))
    ... def foo(x, y):
    ...     return float(x) + float(y)

    Parameters
    ----------
    *args : Union[
                tuple[Callable],
                tuple[list[Validator], Callable],
                tuple[list[Validator], Validator, Callable]
            ]
        Positional arguments.
        - If a single callable is passed, it's wrapped with the signature
        - If two arguments are passed, the first one is a list of validators for the
          arguments and the second one is the callable to wrap
        - If three arguments are passed, the first one is a list of validators for the
          arguments, the second one is a validator for the return value and the third
          one is the callable to wrap
    **kwargs : dict[str, Validator]
        Validators for the arguments.

    Returns
    -------
    Callable
    """
    if _1 is None:
        return functools.partial(annotated, **kwargs)
    elif _2 is None:
        if callable(_1):
            func, validators, return_validator = _1, None, None
        else:
            return functools.partial(annotated, _1, **kwargs)
    elif _3 is None:
        if not isinstance(_2, Validator):
            func, validators, return_validator = _2, _1, None
        else:
            return functools.partial(annotated, _1, _2, **kwargs)
    else:
        func, validators, return_validator = _3, _1, _2

    sig = Signature.from_callable(
        func, validators=validators or kwargs, return_validator=return_validator
    )

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        # 1. Validate the passed arguments
        values = sig.validate(*args, **kwargs)
        # 2. Reconstruction of the original arguments
        args, kwargs = sig.unbind(values)
        # 3. Call the function with the validated arguments
        result = func(*args, **kwargs)
        # 4. Validate the return value
        return sig.validate_return(result, {})

    wrapped.__signature__ = sig

    return wrapped
