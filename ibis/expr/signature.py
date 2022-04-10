from ibis.common.grounds import (  # noqa: F401
    Annotable,
    AnnotableMeta,
    Parameter,
)
from ibis.common.validators import EMPTY, Optional, Validator  # noqa: F401
from ibis.util import all_of, deprecated


class _ValidatorFunction(Validator):
    def __init__(self, fn):
        self.fn = fn

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)


class _InstanceOf(Validator):
    def __init__(self, typ):
        self.typ = typ

    def __call__(self, arg, **kwargs):
        if not isinstance(arg, self.typ):
            raise TypeError(self.typ)
        return arg


@deprecated(version="3.0", instead="use Validator if needed")
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
        validator = _InstanceOf(validator)
    elif isinstance(validator, tuple):
        assert all_of(validator, type)
        validator = _InstanceOf(validator)
    elif isinstance(validator, Validator):
        validator = validator
    elif callable(validator):
        validator = _ValidatorFunction(validator)
    else:
        raise TypeError(
            'Argument validator must be a callable, type or '
            'tuple of types, given: {}'.format(validator)
        )

    if default is EMPTY:
        return validator
    else:
        return Optional(validator, default=default)
