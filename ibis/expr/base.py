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

# Here we are working on the basic data structure for building Ibis expressions
# (an intermediate representation which can be compiled to a target backend,
# e.g. Impala SQL)
#
# The design and class structure here must be explicitly guided by the kind of
# user experience (i.e., highly interactive, suitable for introspection and
# console/notebook use) we want to deliver.
#
# All data structures should be treated as immutable (as much as Python objects
# are immutable; we'll behave as if they are).
#
# Expressions can be parameterized both by tables (known schema, but not bound
# to a particular table in any database), fields, and literal values. In order
# to execute an expression containing parameters, the user must perform a
# actual data. Mixing table and field parameters can lead to tricky binding
# scenarios -- essentially all unbound field parameters within a particular
# table expression must originate from the same concrete table. Internally we
# can identify the "logical tables" in the expression and present those to the
# user for the binding. Good introspection capability will be important
# here. Literal parameters are much simpler. A literal parameter is declared
# and used however many times the user wishes; binding in that case simply
# introduces the actual value to be used.
#
# In some cases, we'll want to be able to indicate that a parameter can either
# be a scalar or array expression. In this case the binding requirements may be
# somewhat more lax.

import ibis.common as com
import ibis.config as config
import ibis.util as util


class Parameter(object):

    """
    Placeholder, to be implemented
    """

    pass


def ir():
    import ibis.expr.types as ir
    return ir

#----------------------------------------------------------------------


class Schema(object):

    """
    Holds table schema information
    """

    def __init__(self, names, types):
        from ibis.expr.types import _validate_type
        if not isinstance(names, list):
            names = list(names)
        self.names = names
        self.types = [_validate_type(x) for x in types]

        self._name_locs = dict((v, i) for i, v in enumerate(self.names))

        if len(self._name_locs) < len(self.names):
            raise com.IntegrityError('Duplicate column names')

    def __repr__(self):
        return self._repr()

    def __len__(self):
        return len(self.names)

    def _repr(self):
        return "%s(%s, %s)" % (type(self).__name__, repr(self.names),
                               repr(self.types))

    def __contains__(self, name):
        return name in self._name_locs

    @classmethod
    def from_tuples(cls, values):
        if len(values):
            names, types = zip(*values)
        else:
            names, types = [], []
        return Schema(names, types)

    @classmethod
    def from_dict(cls, values):
        names = list(values.keys())
        types = values.values()
        return Schema(names, types)

    def equals(self, other):
        return ((self.names == other.names) and
                (self.types == other.types))

    def get_type(self, name):
        return self.types[self._name_locs[name]]

    def append(self, schema):
        names = self.names + schema.names
        types = self.types + schema.types
        return Schema(names, types)



class DataType(object):
    pass


class HasSchema(object):

    """
    Base class representing a structured dataset with a well-defined
    schema.

    Base implementation is for tables that do not reference a particular
    concrete dataset or database table.
    """

    def __init__(self, schema, name=None):
        assert isinstance(schema, Schema)
        self._schema = schema
        self._name = name

    def __repr__(self):
        return self._repr()

    def _repr(self):
        return "%s(%s)" % (type(self).__name__, repr(self.schema))

    @property
    def schema(self):
        return self._schema

    def get_schema(self):
        return self._schema

    def has_schema(self):
        return True

    @property
    def name(self):
        return self._name

    def equals(self, other):
        if type(self) != type(other):
            return False
        return self.schema.equals(other.schema)

    def root_tables(self):
        return [self]


#----------------------------------------------------------------------


class Expr(object):

    """

    """

    def __init__(self, arg):
        # TODO: all inputs must inherit from a common table API
        self._arg = arg

    def __repr__(self):
        if config.options.interactive:
            result = self.execute()
            return repr(result)
        else:
            return self._repr()

    def _repr(self):
        from ibis.expr.format import ExprFormatter
        return ExprFormatter(self).get_result()

    @property
    def _factory(self):
        def factory(arg, name=None):
            return type(self)(arg, name=name)
        return factory

    def execute(self):
        """
        If this expression is based on physical tables in a database backend,
        execute it against that backend.

        Returns
        -------
        result : expression-dependent
          Result of compiling expression and executing in backend
        """
        import ibis.expr.analysis as L
        backend = L.find_backend(self)
        return backend.execute(self)

    def equals(self, other):
        if type(self) != type(other):
            return False
        return self._arg.equals(other._arg)

    def op(self):
        raise NotImplementedError

    def _can_compare(self, other):
        return False

    def _root_tables(self):
        return self.op().root_tables()

    def _get_unbound_tables(self):
        # The expression graph may contain one or more tables of a particular
        # known schema
        pass



class Node(object):

    """
    Node is the base class for all relational algebra and analytical
    functionality. It transforms the input expressions into an output
    expression.

    Each node implementation is responsible for validating the inputs,
    including any type promotion and / or casting issues, and producing a
    well-typed expression

    Note that Node is deliberately not made an expression subclass: think
    of Node as merely a typed expression builder.
    """

    def __init__(self, args):
        self.args = args

    def __repr__(self):
        return self._repr()

    def _repr(self):
        # Quick and dirty to get us started
        opname = type(self).__name__
        pprint_args = [repr(x) for x in self.args]
        return '%s(%s)' % (opname, ', '.join(pprint_args))

    def flat_args(self):
        for arg in self.args:
            if isinstance(arg, (tuple, list)):
                for x in arg:
                    yield x
            else:
                yield arg

    def equals(self, other):
        if type(self) != type(other):
            return False

        if len(self.args) != len(other.args):
            return False

        def is_equal(left, right):
            if isinstance(left, list):
                if not isinstance(right, list):
                    return False
                for a, b in zip(left, right):
                    if not is_equal(a, b):
                        return False
                return True

            if hasattr(left, 'equals'):
                return left.equals(right)
            else:
                return left == right
            return True

        for left, right in zip(self.args, other.args):
            if not is_equal(left, right):
                return False
        return True

    def to_expr(self):
        """
        This function must resolve the output type of the expression and return
        the node wrapped in the appropriate ValueExpr type.
        """
        raise NotImplementedError


class ValueNode(Node):

    def to_expr(self):
        klass = self.output_type()
        return klass(self)

    def _ensure_value(self, expr):
        if not isinstance(expr, ir().ValueExpr):
            raise TypeError('Must be a value, got: %s' % repr(expr))

    def _ensure_array(self, expr):
        if not isinstance(expr, ir().ArrayExpr):
            raise TypeError('Must be an array, got: %s' % repr(expr))

    def _ensure_scalar(self, expr):
        if not isinstance(expr, ir().ScalarExpr):
            raise TypeError('Must be a scalar, got: %s' % repr(expr))

    def root_tables(self):
        return self.arg._root_tables()

    def output_type(self):
        raise NotImplementedError

    def resolve_name(self):
        raise NotImplementedError



class Literal(ValueNode):

    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return 'Literal(%s)' % repr(self.value)

    @property
    def args(self):
        return [self.value]

    def equals(self, other):
        if not isinstance(other, Literal):
            return False
        return (type(self.value) == type(other.value)
                and self.value == other.value)

    def output_type(self):
        import ibis.expr.rules as rules
        if isinstance(self.value, bool):
            klass = ir().BooleanScalar
        elif isinstance(self.value, (int, long)):
            int_type = rules.int_literal_class(self.value)
            klass = ir().scalar_type(int_type)
        elif isinstance(self.value, float):
            klass = ir().DoubleScalar
        elif isinstance(self.value, basestring):
            klass = ir().StringScalar

        return klass

    def root_tables(self):
        return []


class NullLiteral(ValueNode):

    """
    Typeless NULL literal
    """

    def __init__(self):
        return

    @property
    def args(self):
        return [None]

    def equals(other):
        return isinstance(other, NullLiteral)

    def output_type(self):
        return ir().NullScalar

    def root_tables(self):
        return []


class ArrayNode(ValueNode):

    def __init__(self, expr):
        self._ensure_array(expr)
        ValueNode.__init__(self, [expr])

    def output_type(self):
        return NotImplementedError

    def to_expr(self):
        klass = self.output_type()
        return klass(self)


class TableNode(Node):

    def get_type(self, name):
        return self.get_schema().get_type(name)

    def to_expr(self):
        return ir().TableExpr(self)


class Reduction(ArrayNode):

    def __init__(self, arg):
        self.arg = arg
        ArrayNode.__init__(self, arg)

    def root_tables(self):
        return self.arg._root_tables()

    def resolve_name(self):
        return self.arg.get_name()


class BlockingTableNode(TableNode):
    # Try to represent the fact that whatever lies here is a semantically
    # distinct table. Like projections, aggregations, and so forth
    pass




def distinct_roots(*args):
    all_roots = []
    for arg in args:
        all_roots.extend(arg._root_tables())
    return util.unique_by_key(all_roots, id)
