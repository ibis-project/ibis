import inspect
from collections import OrderedDict

import ibis.expr.rules as rlz
import ibis.util as util

try:
    from cytoolz import unique
except ImportError:
    from toolz import unique


_undefined = object()  # marker for missing argument


class Argument:
    """Argument definition."""

    __slots__ = 'validator', 'default', 'show'

    def __init__(self, validator, default=_undefined, show=True):
        """Argument constructor

        Parameters
        ----------
        validator : Union[Callable[[arg], coerced], Type, Tuple[Type]]
            Function which handles validation and/or coercion of the given
            argument.
        default : Union[Any, Callable[[], str]]
            In case of missing (None) value for validation this will be used.
            Note, that default value (except for None) must also pass the inner
            validator.
            If callable is passed, it will be executed just before the inner,
            and itsreturn value will be treaded as default.
        show : bool
            Whether to show this argument in an :class:`~ibis.expr.types.Expr`
            that contains it.
        """
        self.default = default
        self.show = show
        if isinstance(validator, type):
            self.validator = rlz.instance_of(validator)
        elif isinstance(validator, tuple):
            assert util.all_of(validator, type)
            self.validator = rlz.instance_of(validator)
        elif callable(validator):
            self.validator = validator
        else:
            raise TypeError(
                'Argument validator must be a callable, type or '
                'tuple of types, given: {}'.format(validator)
            )

    def __eq__(self, other):
        return (
            self.validator == other.validator and self.default == other.default
        )

    @property
    def optional(self):
        return self.default is not _undefined

    def validate(self, value=_undefined, name=None):
        """
        Parameters
        ----------
        value : Any, default undefined
          Raises TypeError if argument is mandatory but not value has been
          given.
        name : Optional[str]
          Argument name for error message
        """
        if self.optional:
            if value is _undefined or value is None:
                if self.default is None:
                    return None
                elif util.is_function(self.default):
                    value = self.default()
                else:
                    value = self.default
        elif value is _undefined:
            if name is not None:
                name_msg = "argument `{}`".format(name)
            else:
                name_msg = "unnamed argument"
            raise TypeError("Missing required value for {}".format(name_msg))

        return self.validator(value)

    __call__ = validate  # syntactic sugar


class TypeSignature(OrderedDict):

    __slots__ = ()

    @classmethod
    def from_dtypes(cls, dtypes):
        return cls(
            ('_{}'.format(i), Argument(rlz.value(dtype)))
            for i, dtype in enumerate(dtypes)
        )

    def validate(self, *args, **kwargs):
        parameters = [
            inspect.Parameter(
                name,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                default=_undefined,
            )
            for (name, argument) in self.items()
        ]
        sig = inspect.Signature(parameters)
        bindings = sig.bind(*args, **kwargs)

        # The inspect.Parameter objects in parameters all have default
        # value _undefined, which will be bound to all arguments that weren't
        # passed in.
        bindings.apply_defaults()

        result = []
        for (name, arg_value) in bindings.arguments.items():
            argument = self[name]
            # If this arg wasn't passed in: since argument.default has the
            # correct value and _undefined was given as the default for the
            # Parameter object corresponding to this argument, arg_value got
            # the value _undefined when bindings.apply_defaults() was called,
            # so the behavior of argument.validate here is correct.
            value = argument.validate(arg_value, name=name)
            result.append((name, value))

        return result

    __call__ = validate  # syntactic sugar

    def names(self):
        return tuple(self.keys())


class AnnotableMeta(type):
    @classmethod
    def __prepare__(metacls, name, bases, **kwds):
        return OrderedDict()

    def __new__(meta, name, bases, dct):
        slots, signature = [], TypeSignature()

        for parent in bases:
            # inherit parent slots
            if hasattr(parent, '__slots__'):
                slots += parent.__slots__
            # inherit from parent signatures
            if hasattr(parent, 'signature'):
                signature.update(parent.signature)

        # finally apply definitions from the currently created class
        # thanks to __prepare__ attrs are already ordered
        attribs = {}
        for k, v in dct.items():
            if isinstance(v, Argument):
                # so we can set directly
                signature[k] = v
            else:
                attribs[k] = v

        # if slots or signature are defined no inheritance happens
        signature = attribs.get('signature', signature)
        slots = attribs.get('__slots__', tuple(slots)) + signature.names()

        attribs['signature'] = signature
        attribs['__slots__'] = tuple(unique(slots))

        return super().__new__(meta, name, bases, attribs)


class Annotable(metaclass=AnnotableMeta):

    __slots__ = ()

    def __init__(self, *args, **kwargs):
        for name, value in self.signature.validate(*args, **kwargs):
            setattr(self, name, value)
        self._validate()

    def _validate(self):
        pass

    @property
    def args(self):
        return tuple(getattr(self, name) for name in self.signature.names())

    @property
    def argnames(self):
        return self.signature.names()
