import abc
import operator

import ibis
import ibis.expr.operations as ops


class Keyed(object):

    __slots__ = ()

    def __getitem__(self, name):
        return Item(name, self)

    def __getattr__(self, name):
        try:
            return object.__getattribute__(self, name)
        except AttributeError as e:
            if name.startswith('_') and name.endswith('_'):
                raise e
            return Attribute(name, self)


class Operations(object):

    __slots__ = ()

    def __add__(self, other):
        return Add(self, other)

    def __sub__(self, other):
        return Sub(self, other)

    def __mul__(self, other):
        return Mul(self, other)

    def __truediv__(self, other):
        return Div(self, other)

    def __div__(self, other):
        return Div(self, other)

    def __floordiv__(self, other):
        return FloorDiv(self, other)

    def __pow__(self, other):
        return Pow(self, other)

    def __mod__(self, other):
        return Mod(self, other)

    def __eq__(self, other):
        return Eq(self, other)

    def __ne__(self, other):
        return Ne(self, other)

    def __lt__(self, other):
        return Lt(self, other)

    def __le__(self, other):
        return Le(self, other)

    def __gt__(self, other):
        return Gt(self, other)

    def __ge__(self, other):
        return Ge(self, other)


class Value(Keyed, Operations):

    __slots__ = 'name', 'parent'

    def __init__(self, name=None, parent=None):
        self.name = name
        self.parent = parent

    def __hash__(self):
        return hash((self.name, self.parent))


class Binary(Value):

    __slots__ = 'left', 'right'

    def __init__(self, left, right):
        self.left = left
        self.right = right

    def resolve(self, o, scope):
        try:
            left = self.left.resolve(o, scope)
        except AttributeError:
            left = self.left
        try:
            right = self.right.resolve(o, scope)
        except AttributeError:
            right = self.right
        return self.operate(left, right)

    def __call__(self, other):
        return self.resolve(other, {X: other})


class Unary(Value):

    __slots__ = 'operand',

    def __init__(self, operand):
        self.operand = operand

    def resolve(self, operand, scope):
        return self.operate(self.operand.resolve(operand, scope))


class Add(Binary):

    __slots__ = ()

    operate = operator.add


class Sub(Binary):

    __slots__ = ()

    operate = operator.sub


class Mul(Binary):

    __slots__ = ()

    operate = operator.mul


class Div(Binary):

    __slots__ = ()

    operate = operator.truediv


class FloorDiv(Binary):

    __slots__ = ()

    operate = operator.floordiv


class Pow(Binary):

    __slots__ = ()

    operate = operator.pow


class Mod(Binary):

    __slots__ = ()

    operate = operator.mod


class Eq(Binary):

    __slots__ = ()

    operate = operator.eq


class Ne(Binary):

    __slots__ = ()

    operate = operator.ne


class Lt(Binary):

    __slots__ = ()

    operate = operator.lt


class Le(Binary):

    __slots__ = ()

    operate = operator.le


class Gt(Binary):

    __slots__ = ()

    operate = operator.gt


class Ge(Binary):

    __slots__ = ()

    operate = operator.ge


class Getter(Value):

    __slots__ = ()

    def resolve(self, o, scope=None):
        try:
            parent = scope[self.parent]
        except KeyError:
            parent = o
        return parent[self.name]

    def __call__(self, o):
        return self.resolve(o, scope={X: o})


class Attribute(Getter):

    __slots__ = ()

    def __repr__(self):
        return '{0.parent}.{0.name}'.format(self)


class Item(Getter):

    __slots__ = ()

    def __repr__(self):
        return '{0.parent}[{0.name!r}]'.format(self)


class desc(Value):

    __slots__ = ()

    def __init__(self, expr):
        super(desc, self).__init__(parent=expr)

    def resolve(self, o, scope=None):
        return ibis.desc(self.parent.resolve(o, scope=scope))


X = Value('X')
Y = Value('Y')


class Verb(metaclass=abc.ABCMeta):

    __slots__ = ()

    @abc.abstractmethod
    def __call__(self, other):
        pass

    def __rrshift__(self, other):
        return self(other)

    def resolve(self, other, scope):
        return self(other)


class groupby(Verb, Keyed):

    __slots__ = 'keys',

    def __init__(self, *keys):
        self.keys = keys

    def __call__(self, expr):
        return expr.groupby([
            key.resolve(expr, {X: expr}) for key in self.keys
        ])


class select(Verb, Keyed):

    __slots__ = 'columns',

    def __init__(self, *columns):
        self.columns = columns

    def __call__(self, expr):
        op = expr.op()
        if isinstance(op, ops.Join):
            scope = {X: op.left, Y: op.right}
        else:
            scope = {X: expr}
        return expr.projection([
            column.resolve(expr, scope=scope) for column in self.columns
        ])


class sift(Verb, Keyed):

    __slots__ = 'predicates',

    def __init__(self, *predicates):
        self.predicates = predicates

    def __call__(self, expr):
        scope = {X: expr}
        return expr.filter([
            predicate.resolve(expr, scope=scope)
            for predicate in self.predicates
        ])


class summarize(Verb, Keyed):

    __slots__ = 'metrics',

    def __init__(self, **metrics):
        self.metrics = sorted(metrics.items(), key=operator.itemgetter(0))

    def __call__(self, grouped):
        return grouped.aggregate([
            operation(grouped.table).name(name)
            for name, operation in self.metrics
        ])


class head(Verb, Keyed):

    __slots__ = 'n',

    def __init__(self, n=5):
        self.n = n

    def __call__(self, expr):
        return expr.head(self.n)


class Reduction(Verb, Operations):

    __slots__ = 'column', 'where', 'func',

    def __init__(self, column, where=None):
        self.column = column
        self.where = where
        self.func = operator.attrgetter(type(self).__name__.lower())

    def __call__(self, expr):
        where = self.where
        scope = {X: expr}
        column = self.column.resolve(expr, scope)
        return self.func(column)(
            where=where.resolve(expr, scope) if where is not None else where
        )


class mean(Reduction):

    __slots__ = ()


class sum(Reduction):

    __slots__ = ()


class count(Reduction):

    __slots__ = ()


n = count


class SpreadReduction(Reduction):

    __slots__ = 'how',

    def __init__(self, column, where=None, how='sample'):
        super(SpreadReduction, self).__init__(column, where=where)
        self.how = how

    def __call__(self, o):
        where = self.where
        scope = {X: o}
        column = self.column.resolve(o, scope)
        return self.func(column)(
            where=where.resolve(o, scope) if where is not None else where,
            how=self.how
        )


class var(SpreadReduction):

    __slots__ = ()


class std(SpreadReduction):

    __slots__ = ()


class min(Reduction):

    __slots__ = ()


class max(Reduction):

    __slots__ = ()


class mutate(Verb, Keyed):

    __slots__ = 'mutations',

    def __init__(self, **mutations):
        self.mutations = mutations

    def __call__(self, expr):
        return expr.mutate(**{
            name: column.resolve(expr, {X: expr})
            for name, column in self.mutations.items()
        })


class transmute(Verb, Keyed):

    __slots__ = 'mutations',

    def __init__(self, **mutations):
        self.mutations = mutations

    def __call__(self, expr):
        columns = [
            column.name(name) for name, column in self.mutations.items()
        ]
        return expr[columns]


class sort_by(Verb, Keyed):

    __slots__ = 'sort_keys',

    def __init__(self, *sort_keys):
        self.sort_keys = sort_keys

    def __call__(self, expr):
        return expr.sort_by([
            key.resolve(expr, {X: expr}) for key in self.sort_keys
        ])


class On(object):

    __slots__ = 'right', 'on',

    def __init__(self, right, on):
        self.right = right
        self.on = on

    def resolve(self, left, scope=None):
        if isinstance(self.on, Value):
            return self.on.resolve(left, scope=scope)
        else:
            return self.on


class join(Verb, Keyed):

    __slots__ = 'right', 'on', 'how',

    def __init__(self, right, on, how='inner'):
        self.right = right
        self.on = On(right, on)
        self.how = how

    def __call__(self, left):
        right = self.right
        on = self.on.resolve(left, scope={X: left, Y: right})
        return left.join(right, on, how=self.how)


class inner_join(join):

    __slots__ = ()

    def __init__(self, right, on):
        super(inner_join, self).__init__(right, on, how='inner')


class left_join(join):

    __slots__ = ()

    def __init__(self, right, on):
        super(left_join, self).__init__(right, on, how='left')


class right_join(join):

    __slots__ = ()

    def __init__(self, right, on):
        super(right_join, self).__init__(right, on, how='right')


class outer_join(join):

    __slots__ = ()

    def __init__(self, right, on):
        super(outer_join, self).__init__(right, on, how='outer')


class semi_join(join):

    __slots__ = ()

    def __init__(self, right, on):
        super(semi_join, self).__init__(right, on, how='semi')


class anti_join(join):

    __slots__ = ()

    def __init__(self, right, on):
        super(anti_join, self).__init__(right, on, how='anti')


class do(object):

    __slots__ = 'execute',

    def __init__(self, execute=operator.methodcaller('execute')):
        self.execute = execute

    def __call__(self, expr):
        return self.execute(expr)


def from_dataframe(df, name='t'):
    return ibis.pandas.connect({name: df}).table(name)
