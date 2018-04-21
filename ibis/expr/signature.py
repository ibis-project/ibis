import six
import itertools

import ibis.util as util
import ibis.expr.rules as rlz

from ibis.compat import PY2
from collections import OrderedDict

try:
    from cytoolz import unique
except ImportError:
    from toolz import unique


_undefined = object()  # marker for missing argument


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
        default : Union[Any, Callable[[], str]]
          In case of missing (None) value for validation this will be used.
          Note, that default value (except for None) must also pass the inner
          validator.
          If callable is passed, it will be executed just before the inner, and
          itsreturn value will be treaded as default.
        """
        if PY2:
            self._serial = next(self._counter)

        self.default = default
        if isinstance(validator, type):
            self.validator = rlz.instance_of(validator)
        elif isinstance(validator, tuple):
            assert util.all_of(validator, type)
            self.validator = rlz.instance_of(validator)
        elif callable(validator):
            self.validator = validator
        else:
            raise TypeError('Argument validator must be a callable, type or '
                            'tuple of types, given: {}'.format(validator))

    def __eq__(self, other):
        return (
            self.validator == other.validator and
            self.default == other.default
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
                name = ' `{}`'.format(name)
            raise TypeError('Missing required value for argument' + name)

        return self.validator(value)

    __call__ = validate  # syntactic sugar


class TypeSignature(OrderedDict):

    __slots__ = ()

    @classmethod
    def from_dtypes(cls, dtypes):
        return cls(('_{}'.format(i), Argument(rlz.value(dtype)))
                   for i, dtype in enumerate(dtypes))

    def validate(self, *args, **kwargs):
        result = []
        for i, (name, argument) in enumerate(self.items()):
            if i < len(args):
                if name in kwargs:
                    raise TypeError(
                        'Got multiple values for argument {}'.format(name)
                    )
                value = argument.validate(args[i], name=name)
            elif name in kwargs:
                value = argument.validate(kwargs[name], name=name)
            else:
                value = argument.validate(name=name)

            result.append((name, value))

        return result

    __call__ = validate  # syntactic sugar

    def names(self):
        return tuple(self.keys())


class AnnotableMeta(type):

    if PY2:
        @staticmethod
        def _precedes(arg1, arg2):
            """Comparator helper for sorting name-argument pairs"""
            return cmp(arg1[1]._serial, arg2[1]._serial)  # noqa: F821
    else:
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
        if PY2:
            # on python 2 we cannot maintain definition order
            attribs, arguments = {}, []
            for k, v in dct.items():
                if isinstance(v, Argument):
                    arguments.append((k, v))
                else:
                    attribs[k] = v

            # so we need to sort arguments based on their unique counter
            signature.update(sorted(arguments, cmp=meta._precedes))
        else:
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

        return super(AnnotableMeta, meta).__new__(meta, name, bases, attribs)


@six.add_metaclass(AnnotableMeta)
class Annotable(object):

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
