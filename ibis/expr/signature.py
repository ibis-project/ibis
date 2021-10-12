import inspect
from collections import OrderedDict

import ibis.expr.rules as rlz
import ibis.util as util

try:
    from cytoolz import unique
except ImportError:
    from toolz import unique


EMPTY = inspect.Parameter.empty  # marker for missing argument


class Validator:
    def __call__(self, arg, **kwargs):
        raise NotImplementedError()


class ValidatorFunction(Validator):

    __slots__ = ('fn',)

    def __init__(self, fn):
        self.fn = fn

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)


class Optional(Validator):

    __slots__ = ('validator', 'default')

    def __init__(self, validator, default=None):
        self.validator = validator
        self.default = default

    def __call__(self, arg, **kwargs):
        if arg is None:
            if self.default is None:
                return None
            elif util.is_function(self.default):
                arg = self.default()
            else:
                arg = self.default

        return self.validator(arg, **kwargs)


class Parameter(inspect.Parameter):

    __slots__ = ('_validator',)

    def __init__(self, name, *, validator=EMPTY):
        super().__init__(
            name,
            kind=Parameter.POSITIONAL_OR_KEYWORD,
            default=None if isinstance(validator, Optional) else EMPTY,
        )
        self._validator = validator

    @property
    def validator(self):
        return self._validator

    def validate(self, this, arg):
        if self.validator is EMPTY:
            return arg
        else:
            return self.validator(arg, this=this)


def Argument(validator, default=EMPTY):
    if isinstance(validator, Validator):
        pass
    elif isinstance(validator, type):
        validator = rlz.instance_of(validator)
    elif isinstance(validator, tuple):
        assert util.all_of(validator, type)
        validator = rlz.instance_of(validator)
    elif isinstance(validator, Validator):
        validator = validator
    elif callable(validator):
        validator = ValidatorFunction(validator)
    else:
        raise TypeError(
            'Argument validator must be a callable, type or '
            'tuple of types, given: {}'.format(validator)
        )

    if default is EMPTY:
        return validator
    else:
        return Optional(validator, default=default)


class AnnotableMeta(type):
    @classmethod
    def __prepare__(metacls, name, bases, **kwds):
        return OrderedDict()

    def __new__(metacls, clsname, bases, dct):
        slots, params = [], OrderedDict()

        for parent in bases:
            # inherit parent slots
            if hasattr(parent, '__slots__'):
                slots += parent.__slots__
            # inherit from parent signatures
            if hasattr(parent, '__signature__'):
                params.update(parent.__signature__.parameters)

        # finally apply definitions from the currently created class
        # thanks to __prepare__ attrs are already ordered
        attribs = {}
        for name, attrib in dct.items():
            if isinstance(attrib, Validator):
                # so we can set directly
                params[name] = Parameter(name, validator=attrib)
            else:
                attribs[name] = attrib

        # if slots or signature are defined no inheritance happens
        slots = attribs.get('__slots__', tuple(slots))
        slots += tuple(params.keys())

        # mandatory fields without default values must preceed the optional
        # ones in the function signature, the partial ordering will be kept
        params = sorted(
            params.values(), key=lambda p: p.default is EMPTY, reverse=True
        )
        signature = attribs.get('__signature__', inspect.Signature(params))

        attribs['__slots__'] = tuple(unique(slots))
        attribs['__signature__'] = signature

        return super().__new__(metacls, clsname, bases, attribs)


class Annotable(metaclass=AnnotableMeta):

    __slots__ = ()

    def __init__(self, *args, **kwargs):
        bound = self.__signature__.bind(*args, **kwargs)
        bound.apply_defaults()
        for name, value in bound.arguments.items():
            param = self.__signature__.parameters[name]
            setattr(self, name, param.validate(self, value))
        self._validate()

    def _validate(self):
        pass

    @property
    def argnames(self):
        return tuple(self.__signature__.parameters.keys())

    @property
    def args(self):
        return tuple(getattr(self, name) for name in self.argnames)
