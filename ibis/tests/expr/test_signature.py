import pytest

from ibis.expr.signature import Annotable, Argument
from ibis.tests.util import assert_pickle_roundtrip

ARGUMENT_USAGE_MSG = r".*Argument.* is deprecated .* v3\.0; use Validator"

with pytest.warns(FutureWarning, match=ARGUMENT_USAGE_MSG):

    class MagicString(Annotable):
        foo = Argument(str)
        bar = Argument(bool)
        baz = Argument(int)

        def __eq__(self, other):
            return self.args == other.args


def test_argument_raise_on_missing_value():
    with pytest.warns(FutureWarning, match=ARGUMENT_USAGE_MSG):
        validator = Argument(lambda x: x)

    expected_msg = "missing 1 required positional argument"
    with pytest.raises(TypeError, match=expected_msg):
        validator()

    expected_msg = "got an unexpected keyword argument 'name'"
    with pytest.raises(TypeError, match=expected_msg):
        validator(name='mandatory')


def test_argument_is_deprecated():
    with pytest.warns(FutureWarning, match=ARGUMENT_USAGE_MSG):
        Argument(str)


@pytest.mark.parametrize('validator', [3, 'coerce'])
def test_invalid_validator(validator):
    with pytest.raises(TypeError):
        with pytest.warns(FutureWarning, match=ARGUMENT_USAGE_MSG):
            Argument(validator)


def test_invalid_arity_validator():
    with pytest.warns(FutureWarning, match=ARGUMENT_USAGE_MSG):
        arg = Argument(lambda x, y: x + y)

    with pytest.raises(TypeError):
        arg('value')


def test_pickling_support():
    obj = MagicString(foo="magic", bar=True, baz=8)
    assert_pickle_roundtrip(obj)
