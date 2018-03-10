import six
import itertools

import ibis.util as util
from ibis.compat import PY2
from collections import OrderedDict

try:
    from cytoolz import flip, unique
except ImportError:
    from toolz import flip, unique


# TODO: could use the primitives defined here to types too


_undefined = object()  # marker for missing arguments
instance_of = flip(isinstance)


class Argument(object):
    """Argument definition

    """
    if PY2:
        # required to maintain definition order in Annotated metaclass
        _counter = itertools.count()
        __slots__ = '_serial', 'validator', 'default'
    else:
        __slots__ = 'validator', 'default'

    def __init__(self, validator, default=_undefined):
        """Argument constructor

        Parameters
        ----------
        validator : Union[Callable[[arg], coerced], Type, Tuple[Type]]
          Function which handles validation and/or coercion of the given
          argument.
        default : Any
          If given the argument will be optional with the default value given.
        """
        if PY2:
            self._serial = next(self._counter)

        self.default = default
        if callable(validator):
            self.validator = validator
        elif isinstance(validator, type):
            self.validator = instance_of(validator)
        elif isinstance(validator, tuple):
            assert util.all_of(validator, type)
            self.validator = instance_of(validator)
        else:
            raise TypeError('Argument validator must be a callable, type or '
                            'tuple of types, given: {}'.format(validator))

    def __eq__(self, other):
        return (
            self.validator == other.validator and
            self.default == other.default
        )

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
        if value is _undefined:
            if self.default is _undefined:
                if name is not None:
                    name = ' `{}`'.format(name)
                raise TypeError('Missing required value for argument' + name)
            elif callable(self.default):
                return self.default()
            else:
                return self.default

        return self.validator(value)

    __call__ = validate  # syntactic sugar


class Return(object):
    """Acts like a method (output type)"""

    def __call__(self, obj):
        pass


class CallSignature(OrderedDict):

    __slots__ = tuple()

    def validate(self, *args, **kwargs):
        if len(args) > len(self.keys()):
            raise TypeError('takes {} positional arguments ut {} were '
                            'given'.format(len(self.keys()), len(args)))

        result = []
        for i, (name, argument) in enumerate(self.items()):
            if i < len(args):
                if name in kwargs:
                    raise TypeError(
                        'Got multiple values for argument {}'.format(name)
                    )
                value = argument.validate(args[i])
            elif name in kwargs:
                value = argument.validate(kwargs[name])
            else:
                value = argument.validate()

            result.append((name, value))

        return result

    __call__ = validate  # syntactic sugar

    def names(self):
        return tuple(self.keys())


class AnnotableMeta(type):
    """TODO"""

    if PY2:
        @staticmethod
        def _precedes(arg1, arg2):
            """Comparator helper for sorting name-argument pairs"""
            return cmp(arg1[1]._serial, arg2[1]._serial)  # noqa: F821
    else:
        @classmethod
        def __prepare__(metacls, name, bases, **kwds):
            return OrderedDict()

    def __new__(cls, name, parents, attrs):
        slots, signature = [], CallSignature()

        for parent in parents:
            # inherit parent slots
            if hasattr(parent, '__slots__'):
                slots += parent.__slots__
            # inherit from parent signatures
            if hasattr(parent, 'signature'):
                signature.update(parent.signature)

        # finally apply definitions from the currently created class
        if PY2:
            # on python 2 we cannot maintain definition order
            newattrs, arguments = {}, []
            for name, attr in attrs.items():
                if isinstance(attr, Argument):
                    arguments.append((name, attr))
                else:
                    newattrs[name] = attr

            # so we need to sort arguments based on their unique counter
            signature.update(sorted(arguments, cmp=cls._precedes))
        else:
            # thanks to __prepare__ attrs are already ordered
            newattrs = {}
            for name, attr in attrs.items():
                if isinstance(attr, Argument):
                    # so we can set directly
                    signature[name] = attr
                else:
                    newattrs[name] = attr

        # if slots are defined no slot inheritance happens
        slots = newattrs.get('__slots__', tuple(slots)) + signature.names()

        newattrs['signature'] = signature
        newattrs['__slots__'] = tuple(unique(slots))

        return super(AnnotableMeta, cls).__new__(cls, name, parents, newattrs)


class Annotable(six.with_metaclass(AnnotableMeta, object)):

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
