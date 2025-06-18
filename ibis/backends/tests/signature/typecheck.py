"""The following was forked from the python-interface project:

* Copyright (c) 2016-2021, Scott Sanderson

Utilities for typed interfaces.
"""

from __future__ import annotations

from functools import partial
from inspect import Parameter, Signature
from itertools import starmap, takewhile, zip_longest


def valfilter(f, d):
    return {k: v for k, v in d.items() if f(v)}


def dzip(left, right):
    return {k: (left.get(k), right.get(k)) for k in left.keys() & right.keys()}


def complement(f):
    def not_f(*args, **kwargs):
        return not f(*args, **kwargs)

    return not_f


def compatible(
    impl_sig: Signature, iface_sig: Signature, check_annotations: bool = True
) -> bool:
    """Check whether ``impl_sig`` is compatible with ``iface_sig``.

    Parameters
    ----------
    impl_sig
        The signature of the implementation function.
    iface_sig
        The signature of the interface function.
    check_annotations
        Whether to also compare signature annotations (default) vs only parameter names.

    In general, an implementation is compatible with an interface if any valid
    way of passing parameters to the interface method is also valid for the
    implementation.

    Consequently, the following differences are allowed between the signature
    of an implementation method and the signature of its interface definition:

    1. An implementation may add new arguments to an interface iff:
       a. All new arguments have default values.
       b. All new arguments accepted positionally (i.e. all non-keyword-only
          arguments) occur after any arguments declared by the interface.
       c. Keyword-only arguments may be reordered by the implementation.

    2. For type-annotated interfaces, type annotations my differ as follows:
       a. Arguments to implementations of an interface may be annotated with
          a **superclass** of the type specified by the interface.
       b. The return type of an implementation may be annotated with a
          **subclass** of the type specified by the interface.
    """
    # Unwrap to get the underlying inspect.Signature objects.
    return all(
        [
            positionals_compatible(
                takewhile(is_positional, impl_sig.parameters.values()),
                takewhile(is_positional, iface_sig.parameters.values()),
                check_annotations=check_annotations,
            ),
            keywords_compatible(
                valfilter(complement(is_positional), impl_sig.parameters),
                valfilter(complement(is_positional), iface_sig.parameters),
                check_annotations=check_annotations,
            ),
        ]
    )


_POSITIONALS = frozenset([Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD])


def is_positional(arg):
    return arg.kind in _POSITIONALS


def has_default(arg):
    """Does ``arg`` provide a default?."""
    return arg.default is not Parameter.empty


def params_compatible(impl, iface, check_annotations=True):
    if impl is None:
        return False

    if iface is None:
        return has_default(impl)

    checks = (
        impl.name == iface.name
        and impl.kind == iface.kind
        and has_default(impl) == has_default(iface)
    )

    if check_annotations:
        checks = checks and annotations_compatible(impl, iface)

    return checks


def positionals_compatible(impl_positionals, iface_positionals, check_annotations=True):
    params_compat = partial(params_compatible, check_annotations=check_annotations)
    return all(
        starmap(
            params_compat,
            zip_longest(impl_positionals, iface_positionals),
        )
    )


def keywords_compatible(impl_keywords, iface_keywords, check_annotations=True):
    params_compat = partial(params_compatible, check_annotations=check_annotations)
    return all(starmap(params_compat, dzip(impl_keywords, iface_keywords).values()))


def annotations_compatible(impl, iface):
    """Check whether the type annotations of an implementation are compatible with
    the annotations of the interface it implements.
    """
    return impl.annotation == iface.annotation
