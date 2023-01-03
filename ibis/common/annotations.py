from __future__ import annotations

import functools
import inspect
from typing import Any

import toolz

from ibis.common.validators import Validator, any_, option
from ibis.util import DotDict

EMPTY = inspect.Parameter.empty  # marker for missing argument
KEYWORD_ONLY = inspect.Parameter.KEYWORD_ONLY
POSITIONAL_ONLY = inspect.Parameter.POSITIONAL_ONLY
POSITIONAL_OR_KEYWORD = inspect.Parameter.POSITIONAL_OR_KEYWORD
VAR_KEYWORD = inspect.Parameter.VAR_KEYWORD
VAR_POSITIONAL = inspect.Parameter.VAR_POSITIONAL


class Annotation:
    """Base class for all annotations."""

    __slots__ = ('_default', '_validator')

    def __init__(self, validator=None, default=EMPTY):
        self._default = default
        self._validator = validator

    def __eq__(self, other):
        return (
            type(self) is type(other)
            and self._default == other._default
            and self._validator == other._validator
        )

    def validate(self, arg, **kwargs):
        if self._validator is None:
            return arg
        return self._validator(arg, **kwargs)


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
            return None
        elif callable(self._default):
            value = self._default(this)
        else:
            value = self._default
        return self.validate(value, this=this)


class Argument(Annotation):
    """Base class for all fields which should be passed as arguments."""

    @classmethod
    def mandatory(cls, validator=None):
        """Annotation to mark a mandatory argument."""
        return cls(validator)

    @classmethod
    def default(cls, default, validator=None):
        """Annotation to allow missing arguments with a default value."""
        return cls(validator, default=default)

    @classmethod
    def optional(cls, validator=None, default=None):
        """Annotation to allow and treat `None` values as missing arguments."""
        if validator is None:
            validator = option(any_, default=default)
        else:
            validator = option(validator, default=default)
        return cls(validator, default=None)


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
            kind=POSITIONAL_OR_KEYWORD,
            default=annotation._default,
            annotation=annotation._validator,
        )

    def validate(self, arg, *, this):
        if self.annotation is None:
            return arg
        return self.annotation(arg, this=this)


class Signature(inspect.Signature):
    """Validatable signature.

    Primarly used in the implementation of
    ibis.common.grounds.Annotable.
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

        # mandatory fields without default values must preceed the optional
        # ones in the function signature, the partial ordering will be kept
        new_args, new_kwargs = [], []
        inherited_args, inherited_kwargs = [], []

        for name, param in params.items():
            if name in inherited:
                if param.default is EMPTY:
                    inherited_args.append(param)
                else:
                    inherited_kwargs.append(param)
            else:
                if param.default is EMPTY:
                    new_args.append(param)
                else:
                    new_kwargs.append(param)

        return cls(inherited_args + new_args + new_kwargs + inherited_kwargs)

    @classmethod
    def from_callable(cls, fn, validators=None):
        """Create a validateable signature from a callable.

        Parameters
        ----------
        fn : Callable
            Callable to create a signature from.
        validators : dict, default None
            Pass validators for missing type annotations.

        Returns
        -------
        Signature
        """
        sig = super().from_callable(fn)
        params = []
        validators = validators or {}

        for param in sig.parameters.values():
            if param.kind in {
                VAR_POSITIONAL,
                VAR_KEYWORD,
                POSITIONAL_ONLY,
                KEYWORD_ONLY,
            }:
                raise TypeError(f"unsupported parameter kind {param.kind} in {fn}")

            if param.name in validators:
                validator = validators[param.name]
            elif param.annotation is EMPTY:
                validator = any_
            else:
                validator = Validator.from_annotation(
                    param.annotation, module=fn.__module__
                )

            if param.default is EMPTY:
                annot = Argument.mandatory(validator)
            else:
                annot = Argument.default(param.default, validator)

            params.append(Parameter(param.name, annot))

        return cls(params)

    def unbind(self, this: Any):
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
        return {name: getattr(this, name) for name in self.parameters.keys()}

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
            this[name] = param.validate(value, this=this)

        return this


# aliases for convenience
attribute = Attribute
argument = Argument
mandatory = Argument.mandatory
optional = Argument.optional
default = Argument.default


# TODO(kszucs): try to cache validator objects
# TODO(kszucs): try a quicker curry implementation


@toolz.curry
def annotated(fn, **validators):
    """Create functions with arguments validated at runtime.

    There are three ways to apply this decorator:

    1. With type annotations

    >>> @annotated
    ... def foo(x: int, y: str) -> float:
    ...     return float(x) + float(y)

    2. With validators

    >>> from ibis.common.validate import instance_of
    >>> @annotated(x=instance_of(int), y=instance_of(str))
    ... def foo(x, y):
    ...     return float(x) + float(y)

    3. With mixing type annotations and validators where the latter takes precedence

    >>> @annotated(x=instance_of(float))
    ... def foo(x: int, y: str) -> float:
    ...     return x + float(y)

    Parameters
    ----------
    fn : Callable
        Callable to wrap.
    **validators : dict[str, Validator]
        Validators for the arguments.

    Returns
    -------
    Callable
    """
    sig = Signature.from_callable(fn, validators=validators)

    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        kwargs = sig.validate(*args, **kwargs)
        return fn(**kwargs)

    return wrapped
