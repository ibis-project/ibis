from typing import Optional, Union

from ibis.common.typing import evaluate_typehint


def test_evaluate_typehint():
    hint = evaluate_typehint("Union[int, str]", module_name=__name__)
    assert hint == Union[int, str]

    hint = evaluate_typehint(Optional[str], module_name=__name__)
    assert hint == Optional[str]
