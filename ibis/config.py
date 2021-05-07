"""Ibis configuration module."""
# This file has been adapted from pandas/core/config.py. pandas 3-clause BSD
# license. See LICENSES/pandas
#
# Further modifications:
#
# Copyright 2014 Cloudera Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pprint
import re
import warnings
from collections import namedtuple
from contextlib import contextmanager
from typing import Callable

DeprecatedOption = namedtuple('DeprecatedOption', 'key msg rkey removal_ver')
RegisteredOption = namedtuple(
    'RegisteredOption', 'key defval doc validator cb'
)

_deprecated_options = {}  # holds deprecated option metdata
_registered_options = {}  # holds registered option metdata
_global_config = {}  # holds the current values for registered options
_reserved_keys = ['all']  # keys which have a special meaning


class OptionError(AttributeError, KeyError):
    """Exception for ibis.options.

    Backwards compatible with KeyError checks.
    """

    pass


# User API


def _get_single_key(pat, silent):
    keys = _select_options(pat)
    if len(keys) == 0:
        if not silent:
            _warn_if_deprecated(pat)
        raise OptionError('No such keys(s): %r' % pat)
    if len(keys) > 1:
        raise OptionError('Pattern matched multiple keys')
    key = keys[0]

    if not silent:
        _warn_if_deprecated(key)

    key = _translate_key(key)

    return key


def _get_option(pat, silent=False):
    key = _get_single_key(pat, silent)

    # walk the nested dict
    root, k = _get_root(key)
    return root[k]


def _set_option(*args, **kwargs):
    # must at least 1 arg deal with constraints later
    nargs = len(args)
    if not nargs or nargs % 2 != 0:
        raise ValueError(
            "Must provide an even number of non-keyword " "arguments"
        )

    # default to false
    silent = kwargs.get('silent', False)

    for k, v in zip(args[::2], args[1::2]):
        key = _get_single_key(k, silent)

        o = _get_registered_option(key)
        if o and o.validator:
            o.validator(v)

        # walk the nested dict
        root, k = _get_root(key)
        root[k] = v

        if o.cb:
            o.cb(key)


def _describe_option(pat='', _print_desc=True):

    keys = _select_options(pat)
    if len(keys) == 0:
        raise OptionError('No such keys(s)')

    s = ''
    for k in keys:  # filter by pat
        s += _build_option_description(k)

    if _print_desc:
        print(s)
    else:
        return s


def _reset_option(pat, silent=False):

    keys = _select_options(pat)

    if len(keys) == 0:
        raise OptionError('No such keys(s)')

    if len(keys) > 1 and len(pat) < 4 and pat != 'all':
        raise ValueError(
            'You must specify at least 4 characters when '
            'resetting multiple keys, use the special keyword '
            '"all" to reset all the options to their default '
            'value'
        )

    for k in keys:
        _set_option(k, _registered_options[k].defval, silent=silent)


def get_default_val(pat):
    """Return the default value for given pattern.

    Parameters
    ----------
    pat : string

    Returns
    -------
    RegisteredOption (namedtuple) if key is deprecated, None otherwise
    """
    key = _get_single_key(pat, silent=True)
    return _get_registered_option(key).defval


class DictWrapper:
    """Provide attribute-style access to a nested dict."""

    def __init__(self, d, prefix=""):
        object.__setattr__(self, "d", d)
        object.__setattr__(self, "prefix", prefix)

    def __repr__(self):
        """Return the dictionary as formatted string."""
        return pprint.pformat(self.d)

    def __setattr__(self, key, val):
        """Set given value for the given attribute name (key).

        Parameters
        ----------
        key : string
        val : object
        """
        prefix = self.prefix
        if prefix:
            prefix += "."
        prefix += key

        # you can't set new keys and you can't overwrite subtrees

        if key in self.d and not isinstance(self.d[key], dict):
            _set_option(prefix, val)
        else:
            raise OptionError("You can only set the value of existing options")

    def __getattr__(self, key):
        """Get value for the given attribute name.

        Parameters
        ----------
        key : str

        Returns
        -------
        object
        """
        prefix = self.prefix
        if prefix:
            prefix += "."
        prefix += key

        try:
            v = self.d[key]
        except KeyError as e:
            raise AttributeError(*e.args)

        if isinstance(v, dict):
            return DictWrapper(v, prefix)
        else:
            return _get_option(prefix)

    def __dir__(self):
        """Return all dictionary keys sorted."""
        return sorted(self.d.keys())


class CallableDynamicDoc:
    """Convert __doc__ into a property function.

    For user convenience,  we'd like to have the available options described
    in the docstring. For dev convenience we'd like to generate the docstrings
    dynamically instead of maintaining them by hand. To this, we use this
    class which wraps functions inside a callable, and converts
    __doc__ into a property function. The doctsrings below are templates
    using the py2.6+ advanced formatting syntax to plug in a concise list
    of options, and option descriptions.
    """

    def __init__(self, func, doc_tmpl):
        self.__doc_tmpl__ = doc_tmpl
        self.__func__ = func

    def __call__(self, *args, **kwds):
        """Call the the function defined when the object was initialized."""
        return self.__func__(*args, **kwds)

    @property
    def __doc__(self) -> str:
        """Create automatically a documentation using a template.

        Returns
        -------
        string
        """
        opts_desc = _describe_option('all', _print_desc=False)
        opts_list = pp_options_list(list(_registered_options.keys()))
        return self.__doc_tmpl__.format(
            opts_desc=opts_desc, opts_list=opts_list
        )


_get_option_tmpl = """
get_option(pat)
Retrieves the value of the specified option.
Available options:
{opts_list}
Parameters
----------
pat : str
    Regexp which should match a single option.
    Note: partial matches are supported for convenience, but unless you use the
    full option name (e.g. x.y.z.option_name), your code may break in future
    versions if new options with similar names are introduced.
Returns
-------
result : the value of the option
Raises
------
OptionError : if no such option exists
Notes
-----
The available options with its descriptions:
{opts_desc}
"""

_set_option_tmpl = """
set_option(pat, value)
Sets the value of the specified option.
Available options:
{opts_list}
Parameters
----------
pat : str
    Regexp which should match a single option.
    Note: partial matches are supported for convenience, but unless you use the
    full option name (e.g. x.y.z.option_name), your code may break in future
    versions if new options with similar names are introduced.
value :
    new value of option.
Returns
-------
None
Raises
------
OptionError if no such option exists
Notes
-----
The available options with its descriptions:
{opts_desc}
"""

_describe_option_tmpl = """
describe_option(pat, _print_desc=False)
Prints the description for one or more registered options.
Call with not arguments to get a listing for all registered options.
Available options:
{opts_list}
Parameters
----------
pat : str
    Regexp pattern. All matching keys will have their description displayed.
_print_desc : bool, default True
    If True (default) the description(s) will be printed to stdout.
    Otherwise, the description(s) will be returned as a unicode string
    (for testing).
Returns
-------
None by default, the description(s) as a unicode string if _print_desc
is False
Notes
-----
The available options with its descriptions:
{opts_desc}
"""

_reset_option_tmpl = """
reset_option(pat)
Reset one or more options to their default value.
Pass "all" as argument to reset all options.
Available options:
{opts_list}
Parameters
----------
pat : str/regex
    If specified only options matching `prefix*` will be reset.
    Note: partial matches are supported for convenience, but unless you
    use the full option name (e.g. x.y.z.option_name), your code may break
    in future versions if new options with similar names are introduced.
Returns
-------
None
Notes
-----
The available options with its descriptions:
{opts_desc}
"""

# bind the functions with their docstrings into a Callable
# and use that as the functions exposed in pd.api
get_option = CallableDynamicDoc(_get_option, _get_option_tmpl)
set_option = CallableDynamicDoc(_set_option, _set_option_tmpl)
reset_option = CallableDynamicDoc(_reset_option, _reset_option_tmpl)
describe_option = CallableDynamicDoc(_describe_option, _describe_option_tmpl)
options = DictWrapper(_global_config)

#
# Functions for use by pandas developers, in addition to User - api


class option_context:
    """
    Context manager to temporarily set options in the `with` statement context.

    You need to invoke as ``option_context(pat, val, [(pat, val), ...])``.

    Examples
    --------
    >>> with option_context('interactive', True):
    ...     print(options.interactive)
    True
    >>> options.interactive
    False
    """

    def __init__(self, *args):
        if not (len(args) % 2 == 0 and len(args) >= 2):
            raise ValueError(
                'Need to invoke as'
                'option_context(pat, val, [(pat, val), ...)).'
            )

        self.ops = list(zip(args[::2], args[1::2]))

    def __enter__(self):
        """Create a backup of current options and define new ones."""
        undo = []
        for pat, val in self.ops:
            undo.append((pat, _get_option(pat, silent=True)))

        self.undo = undo

        for pat, val in self.ops:
            _set_option(pat, val, silent=True)

    def __exit__(self, *args):
        """Rollback the options values defined before `with` statement."""
        if self.undo:
            for pat, val in self.undo:
                _set_option(pat, val, silent=True)


def register_option(key, defval, doc='', validator=None, cb=None):
    """Register an option in the package-wide ibis config object.

    Parameters
    ----------
    key
        a fully-qualified key, e.g. "x.y.option - z".
    defval
        the default value of the option
    doc
        a string description of the option
    validator
        a function of a single argument, should raise `ValueError` if
        called with a value which is not a legal value for the option.
    cb
        a function of a single argument "key", which is called
        immediately after an option value is set/reset. key is
        the full name of the option.

    Raises
    ------
    ValueError if `validator` is specified and `defval` is not a valid value.
    """
    import keyword
    import tokenize

    key = key.lower()

    if key in _registered_options:
        raise OptionError("Option '%s' has already been registered" % key)
    if key in _reserved_keys:
        raise OptionError("Option '%s' is a reserved key" % key)

    # the default value should be legal
    if validator:
        validator(defval)

    # walk the nested dict, creating dicts as needed along the path
    path = key.split('.')

    for k in path:
        if not bool(re.match('^' + tokenize.Name + '$', k)):
            raise ValueError("%s is not a valid identifier" % k)
        if keyword.iskeyword(k):
            raise ValueError("%s is a python keyword" % k)

    cursor = _global_config
    for i, p in enumerate(path[:-1]):
        if not isinstance(cursor, dict):
            raise OptionError(
                "Path prefix to option '%s' is already an option"
                % '.'.join(path[:i])
            )
        if p not in cursor:
            cursor[p] = {}
        cursor = cursor[p]

    if not isinstance(cursor, dict):
        raise OptionError(
            "Path prefix to option '%s' is already an option"
            % '.'.join(path[:-1])
        )

    cursor[path[-1]] = defval  # initialize

    # save the option metadata
    _registered_options[key] = RegisteredOption(
        key=key, defval=defval, doc=doc, validator=validator, cb=cb
    )


def deprecate_option(
    key: str, msg: str = None, rkey: str = None, removal_ver: str = None
):
    """Mark option `key` as deprecated.

    If code attempts to access this option, a warning will be produced,
    using `msg` if given, or a default message if not.

    if `rkey` is given, any access to the key will be re-routed to `rkey`.
    Neither the existence of `key` nor that if `rkey` is checked. If they
    do not exist, any subsequence access will fail as usual, after the
    deprecation warning is given.

    Parameters
    ----------
    key : string
        the name of the option to be deprecated. must be a fully-qualified
        option name (e.g "x.y.z.rkey").
    msg : string, optional
        a warning message to output when the key is referenced.
        if no message is given a default message will be emitted.
    rkey : string, optional
        the name of an option to reroute access to.
        If specified, any referenced `key` will be re-routed to `rkey`
        including set/get/reset. rkey must be a fully-qualified option name
        (e.g "x.y.z.rkey"). used by the default message if no `msg` is
        specified.
    removal_ver : string, optional
        specifies the version in which this option will be removed. used by
        the default message if no `msg` is specified.

    Raises
    ------
    OptionError - if key has already been deprecated.
    """
    key = key.lower()

    if key in _deprecated_options:
        raise OptionError(
            "Option '%s' has already been defined as deprecated." % key
        )

    _deprecated_options[key] = DeprecatedOption(key, msg, rkey, removal_ver)


# functions internal to the module


def _select_options(pat: str) -> list:
    """Return a list of keys matching `pat`.

    Parameters
    ----------
    pat : string

    if pat=="all", returns all registered options

    Returns
    -------
    list
    """
    # short-circuit for exact key
    if pat in _registered_options:
        return [pat]

    # else look through all of them
    keys = sorted(_registered_options.keys())
    if pat == 'all':  # reserved key
        return keys

    return [k for k in keys if re.search(pat, k, re.I)]


def _get_root(key: str) -> tuple:
    """Return the parent node of an option.

    Parameters
    ----------
    key : string

    Returns
    -------
    tuple
    """
    path = key.split('.')
    cursor = _global_config
    for p in path[:-1]:
        cursor = cursor[p]
    return cursor, path[-1]


def _is_deprecated(key: str) -> bool:
    """Check if the option is deprecated.

    Parameters
    ----------
    key : string

    Returns
    -------
    bool
        Return True if the given option has been deprecated
    """
    key = key.lower()
    return key in _deprecated_options


def _get_deprecated_option(key: str):
    """
    Retrieve the metadata for a deprecated option, if `key` is deprecated.

    Parameters
    ----------
    key : string

    Returns
    -------
    DeprecatedOption (namedtuple) if key is deprecated, None otherwise
    """
    try:
        d = _deprecated_options[key]
    except KeyError:
        return None
    else:
        return d


def _get_registered_option(key: str):
    """
    Retrieve the option metadata if `key` is a registered option.

    Parameters
    ----------
    key : string

    Returns
    -------
    RegisteredOption (namedtuple) if key is deprecated, None otherwise
    """
    return _registered_options.get(key)


def _translate_key(key: str):
    """Translate a key if necessary.

    If key id deprecated and a replacement key defined, will return the
    replacement key, otherwise returns `key` as - is

    Parameters
    ----------
    key : string
    """
    d = _get_deprecated_option(key)
    if d:
        return d.rkey or key
    else:
        return key


def _warn_if_deprecated(key):
    """
    Check if `key` is a deprecated option and if so, prints a warning.

    Returns
    -------
    bool
        True if `key` is deprecated, False otherwise.
    """
    d = _get_deprecated_option(key)
    if d:
        if d.msg:
            print(d.msg)
            warnings.warn(d.msg, DeprecationWarning)
        else:
            msg = "'%s' is deprecated" % key
            if d.removal_ver:
                msg += ' and will be removed in %s' % d.removal_ver
            if d.rkey:
                msg += ", please use '%s' instead." % d.rkey
            else:
                msg += ', please refrain from using it.'

            warnings.warn(msg, DeprecationWarning)
        return True
    return False


def _build_option_description(k: str) -> str:
    """Build a formatted description of a registered option and prints it.

    Parameters
    ----------
    k : string

    Returns
    -------
    str
    """
    o = _get_registered_option(k)
    d = _get_deprecated_option(k)

    buf = ['{} '.format(k)]

    if o.doc:
        doc = '\n'.join(o.doc.strip().splitlines())
    else:
        doc = 'No description available.'

    buf.append(doc)

    if o:
        buf.append(
            '\n    [default: {}] [currently: {}]'.format(
                o.defval, _get_option(k, True)
            )
        )

    if d:
        buf.append(
            '\n    (Deprecated{})'.format(
                ', use `{}` instead.'.format(d.rkey) if d.rkey else ''
            )
        )

    buf.append('\n\n')
    return ''.join(buf)


def pp_options_list(keys: str, width: int = 80, _print: bool = False) -> str:
    """
    Build a concise listing of available options, grouped by prefix.

    Parameters
    ----------
    keys : string
    width : int
    _print : bool

    Returns
    -------
    string
    """
    from itertools import groupby
    from textwrap import wrap

    def pp(name, ks):
        pfx = '- ' + name + '.[' if name else ''
        ls = wrap(
            ', '.join(ks),
            width,
            initial_indent=pfx,
            subsequent_indent='  ',
            break_long_words=False,
        )
        if ls and ls[-1] and name:
            ls[-1] = ls[-1] + ']'
        return ls

    ls = []
    singles = [x for x in sorted(keys) if x.find('.') < 0]
    if singles:
        ls += pp('', singles)
    keys = [x for x in keys if x.find('.') >= 0]

    for k, g in groupby(sorted(keys), lambda x: x[: x.rfind('.')]):
        ks = [x[len(k) + 1 :] for x in list(g)]
        ls += pp(k, ks)
    s = '\n'.join(ls)
    if _print:
        print(s)
    else:
        return s


# helpers


@contextmanager
def config_prefix(prefix):
    """Create a Context Manager for multiple invocations using a common prefix.

    Context Manager for multiple invocations of API  with a common prefix
    supported API functions: (register / get / set )__option

    Warning: This is not thread - safe, and won't work properly if you import
    the API functions into your module using the "from x import y" construct.

    Examples
    --------
    import ibis.config as cf
    with cf.config_prefix("display.font"):
        cf.register_option("color", "red")
        cf.register_option("size", " 5 pt")
        cf.set_option(size, " 6 pt")
        cf.get_option(size)
        ...
        etc'

    will register options "display.font.color", "display.font.size", set the
    value of "display.font.size"... and so on.
    """
    # Note: reset_option relies on set_option, and on key directly
    # it does not fit in to this monkey-patching scheme
    global register_option, get_option, set_option, reset_option

    def wrap(func):
        def inner(key, *args, **kwds):
            pkey = '%s.%s' % (prefix, key)
            return func(pkey, *args, **kwds)

        return inner

    _register_option = register_option
    _get_option = get_option
    _set_option = set_option

    set_option = wrap(set_option)
    get_option = wrap(get_option)
    register_option = wrap(register_option)

    yield None

    set_option = _set_option
    get_option = _get_option
    register_option = _register_option


# These factories and methods are handy for use as the validator
# arg in register_option


def is_type_factory(_type) -> Callable:
    """Create a function that checks the type of an object.

    The function returned check if the type of a given object is the same of
    the type `_type` given to the function factory.

    Parameters
    ----------
    _type
        a type to be compared against (e.g. type(x) == `_type`)

    Returns
    -------
    validator : function
        a function of a single argument x , which returns the
        True if type(x) is equal to `_type`.
    """
    # checking function
    def inner(x):
        """Check if the type of a given object is equals to the given type."""
        if type(x) != _type:
            raise ValueError("Value must have type '%s'" % str(_type))

    return inner


def is_instance_factory(_type) -> Callable:
    """Create a function that checks if an object is instance of a given type.

    Parameters
    ----------
    `_type` - the type to be checked against

    Returns
    -------
    validator : function
        a function of a single argument x , which returns the
        True if x is an instance of `_type`
    """
    if isinstance(_type, (tuple, list)):
        _type = tuple(_type)
        type_repr = "|".join(map(str, _type))
    else:
        type_repr = "'%s'" % _type

    def inner(x):
        if not isinstance(x, _type):
            raise ValueError("Value must be an instance of %s" % type_repr)

    return inner


def is_one_of_factory(legal_values: list) -> Callable:
    """
    Create a function that check if a given value is in the given list.

    Parameters
    ----------
    legal_values : list

    Returns
    -------
    validator : function
    """
    # checking function
    def inner(x):
        """
        Check if the given value is in the list given to the factory function.

        Parameters
        ----------
        x : object

        Raises
        ------
        ValueError
            If the given value is not in the list given to the factory
            function.
        """
        if x not in legal_values:
            pp_values = map(str, legal_values)
            raise ValueError(
                "Value must be one of %s" % str("|".join(pp_values))
            )

    return inner


# common type validators, for convenience
# usage: register_option(... , validator = is_int)
is_int = is_type_factory(int)
is_bool = is_type_factory(bool)
is_float = is_type_factory(float)
is_str = is_type_factory(str)
is_text = is_instance_factory((str, bytes))
