import copy
import inspect
from typing import Any, Callable, Dict

import ibis.expr.rules as rlz
from ibis import util

EMPTY = inspect.Parameter.empty  # marker for missing argument


class Validator(Callable):
    pass


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
        self.default = copy.deepcopy(default)

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
    """
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
    def __new__(metacls, clsname, bases, dct):
        params = {}
        for parent in bases:
            # inherit from parent signatures
            if hasattr(parent, '__signature__'):
                params.update(parent.__signature__.parameters)

        slots = list(dct.pop('__slots__', []))
        attribs = {}
        for name, attrib in dct.items():
            if isinstance(attrib, Validator):
                # so we can set directly
                params[name] = Parameter(name, validator=attrib)
                slots.append(name)
            else:
                attribs[name] = attrib

        # mandatory fields without default values must preceed the optional
        # ones in the function signature, the partial ordering will be kept
        params = sorted(
            params.values(), key=lambda p: p.default is EMPTY, reverse=True
        )

        attribs['__slots__'] = tuple(slots)
        attribs['__signature__'] = inspect.Signature(params)

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

    def __eq__(self, other) -> bool:
        if self is other:
            return True
        return type(self) == type(other) and self.args == other.args

    def __getstate__(self) -> Dict[str, Any]:
        return {
            key: getattr(self, key)
            for key in self.__signature__.parameters.keys()
        }

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """
        Parameters
        ----------
        state: Dict[str, Any]
            A dictionary storing the objects attributes.
        """
        for key, value in state.items():
            setattr(self, key, value)

    @property
    def argnames(self):
        return tuple(self.__signature__.parameters.keys())

    @property
    def args(self):
        return tuple(getattr(self, name) for name in self.argnames)
