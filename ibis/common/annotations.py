from __future__ import annotations

import inspect
from typing import Any

from ibis.util import DotDict, is_function

EMPTY = inspect.Parameter.empty  # marker for missing argument
VAR_POSITIONAL = inspect.Parameter.VAR_POSITIONAL
POSITIONAL_OR_KEYWORD = inspect.Parameter.POSITIONAL_OR_KEYWORD
KEYWORD_ONLY = inspect.Parameter.KEYWORD_ONLY


def _noop(arg, **kwargs):
    return arg


class Annotation:
    """Base class for all annotations."""

    __slots__ = ()


class Attribute(Annotation):
    """Annotation to mark a field in a class.

    An optional validator can be provider to validate the field every time it
    is set.

    Parameters
    ----------
    validator : Validator, default noop
        Validator to validate the field.
    """

    __slots__ = ('validator',)

    def __init__(self, validator=_noop):
        self.validator = validator

    def __eq__(self, other):
        return type(self) is type(other) and self.validator == other.validator

    def __repr__(self):
        return f"{self.__class__.__name__}({self.validator!r})"

    def validate(self, arg, **kwargs):
        return self.validator(arg, **kwargs)


class Initialized(Attribute):
    """Annotation to initialize an attribute with a callable.

    The callable is called et the end of the instantiation with the instance as
    the only argument. The return value of the callable is assigned to the
    field.

    Parameters
    ----------
    initializer : Callable
        Callable to initialize the field, similar to a property.
    validator : Validator, default noop
        Validator to validate the field every time it is set.
    """

    __slots__ = ('initializer',)

    def __init__(self, initializer, validator=_noop):
        super().__init__(validator)
        self.initializer = initializer

    def __eq__(self, other):
        return super().__eq__(other) and self.initializer == other.initializer

    def initialize(self, this):
        value = self.initializer(this)
        return self.validator(value, this=this)


class Argument(Attribute):
    """Base class for all fields which should be passed as arguments."""

    __slots__ = ()


class Mandatory(Argument):
    """Annotation to mark a mandatory argument."""

    __slots__ = ()


class Default(Argument):
    """Annotation to allow missing arguments with a default value.

    Parameters
    ----------
    default : Any
        Value to return with in case of a missing argument.
    validator : Validator, default noop
        Used to do the actual validation if the argument gets passed.
    """

    __slots__ = ('default',)

    def __init__(self, validator=_noop, default=None):
        super().__init__(validator)
        self.default = default

    def __repr__(self):
        clsname = self.__class__.__name__
        return f"{clsname}({self.validator!r}, default={self.default!r})"

    def __eq__(self, other):
        return super().__eq__(other) and self.default == other.default


class Optional(Default):
    """Annotation to allow and treat `None` values as missing arguments.

    Parameters
    ----------
    validator : Validator
        Used to do the actual validation if the argument gets passed.
    default : Any, default None
        Value to return with in case of a missing argument.
    """

    __slots__ = ()

    def validate(self, arg, **kwargs):
        if arg is None:
            if self.default is None:
                return None
            elif is_function(self.default):
                arg = self.default()
            else:
                arg = self.default
        return super().validate(arg, **kwargs)


class Variadic(Argument):
    """Marker class for validating variadic arguments."""

    __slots__ = ()

    def validate(self, arg, **kwargs):
        return tuple(self.validator(item, **kwargs) for item in arg)


class Parameter(inspect.Parameter):
    """Augmented Parameter class to additionally hold a validator object."""

    __slots__ = ()

    def __init__(self, name, annotation, keyword=False):
        kind = KEYWORD_ONLY if keyword else POSITIONAL_OR_KEYWORD
        default = EMPTY

        if isinstance(annotation, Mandatory):
            pass
        elif isinstance(annotation, Variadic):
            kind = VAR_POSITIONAL
        elif isinstance(annotation, Optional):
            # Note, that this branch shouldn't be necessary since the
            # `Default` branch would handle it, but we support callable
            # defaults for `Optional` arguments which we only check if the
            # passed value is None, otherwise functions passed as values
            # wouldn't work.
            default = None
        elif isinstance(annotation, Default):
            default = annotation.default
        elif not isinstance(annotation, Argument):
            raise TypeError(
                f"Invalid annotation type: {type(annotation).__name__}"
            )

        super().__init__(
            name, kind=kind, default=default, annotation=annotation
        )

    def validate(self, arg, *, this):
        return self.annotation.validate(arg, this=this)


class Signature(inspect.Signature):
    """Validatable signature.

    Primarly used in the implementation of
    ibis.common.grounds.Annotable.
    """

    __slots__ = ()

    @classmethod
    def merge(cls, *signatures, **annotations):
        params = {}
        inherited = set()
        is_variadic = False
        for sig in signatures:
            for name, param in sig.parameters.items():
                is_variadic |= param.kind == VAR_POSITIONAL
                params[name] = param
                inherited.add(name)

        for name, annot in annotations.items():
            param = Parameter(name, annotation=annot, keyword=is_variadic)
            is_variadic |= param.kind == VAR_POSITIONAL
            params[name] = param

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

    def unbind(self, this: Any):
        """Reverse bind of the parameters.

        Attempts to reconstructs the original arguments as positional and
        keyword arguments. Since keyword arguments are the preferred, the
        positional arguments are filled only if the signature has variadic
        args.

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
        args, kwargs = tuple(), {}

        for name, param in self.parameters.items():
            value = getattr(this, name)
            if param.kind == VAR_POSITIONAL:
                args = value
            else:
                kwargs[name] = value

        return args, kwargs

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
default = Default
immutable_property = Initialized
initialized = Initialized
mandatory = Mandatory
optional = Optional
variadic = Variadic


# TODO(kszucs): try to cache validator objects
# TODO(kszucs): try a quicker curry implementation
