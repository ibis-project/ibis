import collections
import enum
import functools
import importlib
import operator
import pathlib
import sys
import tempfile

import flatbuffers
import numpy as np
import pandas as pd

import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.schema as sch
import ibis.expr.types as ir
import ibis.util as util

# ugly side effect, we can probably do better
_computeir_path = pathlib.Path(
    # FIXME(kszucs): read from env variable
    "/Users/kszucs/Workspace/arrow/experimental/computeir"
)
sys.path.append(str(_computeir_path))


class FlatbufferType:
    # TODO(kszucs): consider to use a metaclass instead

    __slots__ = '_name', '_module', '_builder'

    def __init__(self, name, module, builder):
        self._name = name
        self._module = module
        self._builder = builder

    def __repr__(self):
        return f"<FlatbufferType: {self._module.__name__}>"

    def __getattr__(self, field):
        klass = getattr(self._module, self._name)
        return getattr(klass, field)

    def _create_string(self, value):
        return self._builder.CreateString(value)

    def _create_vector(self, name, value):
        assert isinstance(value, list)

        camel = name.title().replace('_', '')
        start = getattr(self._module, f"Start{camel}Vector")
        start(self._builder, len(value))

        for v in reversed(value):
            self._builder.PrependUOffsetTRelative(v)

        return self._builder.EndVector()

    def _convert_arguments(self, kwargs):
        result = {}
        for name, value in kwargs.items():
            if value is None:
                continue
            elif isinstance(value, str):
                value = self._create_string(value)
            elif isinstance(value, list):
                value = self._create_vector(name, value)
            result[name] = value
        return result

    def __call__(self, **kwargs):
        kwargs = self._convert_arguments(kwargs)
        self._module.Start(self._builder)
        for name, value in kwargs.items():
            camel = name.title().replace('_', '')
            setter = getattr(self._module, f"Add{camel}")
            setter(self._builder, value)
        return self._module.End(self._builder)


class FlatbufferNamespace:

    __slots__ = '_namespace', '_builder'

    # TODO(kszucs): perhaps a class level cache for imported modules

    def __init__(self, namespace, builder):
        self._builder = builder
        self._namespace = namespace

    def __repr__(self):
        return f"<FlatbufferNamespace: {self._namespace.__name__}>"

    def __getattr__(self, name):
        module = importlib.import_module(f"{self._namespace}.{name}")
        return FlatbufferType(name, module=module, builder=self._builder)


class PyArrowCompiler:
    def __init__(self, function_names):
        self._names = function_names

    def name_for(self, op):
        return self._names[type(op)]

    def reset(self):
        self._builder = flatbuffers.Builder(0)
        self._namespace_format = FlatbufferNamespace(
            "org.apache.arrow.flatbuf", builder=self._builder
        )
        self._namespace_computeir = FlatbufferNamespace(
            "org.apache.arrow.computeir.flatbuf", builder=self._builder
        )

    @property
    def format(self):
        return self._namespace_format

    @property
    def ir(self):
        return self._namespace_computeir

    def serialize(self, expr):
        self.reset()
        root = translate(expr, compiler=self)
        self._builder.Finish(root)
        return self._builder.Output()


# sum, avg, count, selection, interval


@functools.singledispatch
def translate(*args, **kwargs):
    raise NotImplementedError(args, kwargs)


@translate.register(dt.Integer)
def integer(dtype, name, format):
    bit_width = dtype._nbytes * 8
    is_signed = isinstance(dtype, dt.SignedInteger)
    return format.Field(
        name=name,
        type_type=format.Type.Int,
        type=format.Int(bit_width=bit_width, is_signed=is_signed),
        nullable=dtype.nullable,
    )


@translate.register(dt.Floating)
def floating(dtype, name, format):
    precisions = {
        2: format.Precision.HALF,
        4: format.Precision.SINGLE,
        8: format.Precision.DOUBLE,
    }
    return format.Field(
        name=name,
        type_type=format.Type.FloatingPoint,
        type=format.FloatingPoint(precision=precisions[dtype._nbytes]),
        nullable=dtype.nullable,
    )


@translate.register(dt.String)
def string(dtype, name, format):
    return format.Field(
        name=name,
        type_type=format.Type.Utf8,
        type=format.Utf8(),
        nullable=dtype.nullable,
    )


@translate.register(dt.Binary)
def binary(dtype, name, format):
    return format.Field(
        name=name,
        type_type=format.Type.Binary,
        type=format.Binary(),
        nullable=dtype.nullable,
    )


@translate.register(dt.Date)
def date(dtype, name, format):
    return format.Field(
        name=name,
        type_type=format.Type.Date,
        type=format.Date(unit=format.DateUnit.MILLISECOND),
        nullable=dtype.nullable,
    )


@translate.register(dt.Time)
def time(dtype, name, format):
    return format.Field(
        name=name,
        type_type=format.Type.Time,
        type=format.Time(
            unit=format.TimeUnit.NANOSECOND,
            bit_width=64,
        ),
        nullable=dtype.nullable,
    )


@translate.register(dt.Timestamp)
def timestamp(dtype, name, format):
    return format.Field(
        name=name,
        type_type=format.Type.Timestamp,
        type=format.Timestamp(
            unit=format.TimeUnit.NANOSECOND,
            timezone=dtype.timezone,
        ),
        nullable=dtype.nullable,
    )


@translate.register(dt.Timestamp)
def timestamp(dtype, name, format):
    return format.Field(
        name=name,
        type_type=format.Type.Timestamp,
        type=format.Timestamp(
            unit=format.TimeUnit.NANOSECOND,
            timezone=dtype.timezone,
        ),
        nullable=dtype.nullable,
    )


@translate.register(dt.Interval)
def interval(dtype, name, format):
    # Note: DAY_TIME unit is not required for full arrow compatibility so
    # prefer using MONTH_DAY_NANO instead
    units = {
        'Y': format.IntervalUnit.YEAR_MONTH,
        'Q': format.IntervalUnit.YEAR_MONTH,
        'M': format.IntervalUnit.YEAR_MONTH,
        'W': format.IntervalUnit.MONTH_DAY_NANO,
        'D': format.IntervalUnit.MONTH_DAY_NANO,
        'h': format.IntervalUnit.MONTH_DAY_NANO,
        'm': format.IntervalUnit.MONTH_DAY_NANO,
        's': format.IntervalUnit.MONTH_DAY_NANO,
        'ms': format.IntervalUnit.MONTH_DAY_NANO,
        'us': format.IntervalUnit.MONTH_DAY_NANO,
        'ns': format.IntervalUnit.MONTH_DAY_NANO,
    }
    return format.Field(
        name=name,
        type_type=format.Type.Interval,
        type=format.Interval(unit=units[dtype.unit]),
        nullable=dtype.nullable,
    )


@translate.register(dt.Array)
def array(dtype, name, format):
    return format.Field(
        name=name,
        type_type=format.Type.List,
        type=format.List(),
        nullable=dtype.nullable,
        children=[translate(dtype.value_type, None, format)],
    )


@translate.register(sch.Schema)
def schema(schema, compiler):
    fields = [
        translate(dtype, name, compiler.format)
        for name, dtype in schema.items()
    ]
    return compiler.format.Schema(fields=fields)


@translate.register(ir.Expr)
def expr(expr, compiler):
    return translate(expr.op(), expr, compiler)


@translate.register(ops.Literal)
def literal(op, expr, compiler):
    name = f"{op.dtype.name}Literal"
    klass = getattr(compiler.ir, name)
    return compiler.ir.Expression(
        impl=compiler.ir.Literal(
            impl=klass(value=op.value),
            impl_type=getattr(compiler.ir.LiteralImpl, name),
            type=translate(op.dtype, None, compiler.format),
        ),
        impl_type=compiler.ir.ExpressionImpl.Literal,
    )


@translate.register(ops.BinaryOp)
def binop(op, expr, compiler):
    left = translate(op.left, compiler)
    right = translate(op.right, compiler)

    return compiler.ir.Expression(
        impl=compiler.ir.Call(
            name=compiler.name_for(op), arguments=[left, right]
        ),
        impl_type=compiler.ir.ExpressionImpl.Call,
    )


@translate.register(ops.Sum)
@translate.register(ops.Mean)
def reduction(op, expr, compiler):
    arg = translate(op.arg, compiler)
    return compiler.ir.Expression(
        impl=compiler.ir.Call(name=compiler.name_for(op), arguments=[arg]),
        impl_type=compiler.ir.ExpressionImpl.Call,
    )


@translate.register(ops.SortKey)
def sort_key(op, expr, compiler):
    if op.ascending:
        ordering = compiler.ir.Ordering.NULLS_THEN_ASCENDING
    else:
        ordering = compiler.ir.Ordering.NULLS_THEN_DESCENDING
    return compiler.ir.SortKey(
        expression=translate(op.expr, compiler), ordering=ordering
    )


@translate.register(ops.TableColumn)
def table_column(op, expr, compiler):
    # TODO(kszucs): locate the position of the field from op.table.schema
    schema = op.table.schema()
    position = schema.names.index(op.name)
    return compiler.ir.Expression(
        impl=compiler.ir.FieldRef(
            ref=compiler.ir.FieldIndex(position=position),
            ref_type=compiler.ir.Deref.FieldIndex,
        ),
        impl_type=compiler.ir.ExpressionImpl.FieldRef,
    )


@translate.register(ops.UnboundTable)
def unbound_table(op, expr, compiler):
    return compiler.ir.Relation(
        impl_type=compiler.ir.RelationImpl.Source,
        impl=compiler.ir.Source(
            name=op.name,
            base=compiler.ir.RelBase(
                output_mapping=compiler.ir.PassThrough(),
                output_mapping_type=compiler.ir.Emit.PassThrough,
            ),
            schema=translate(op.schema, compiler),
        ),
    )


@translate.register(ops.Selection)
def selection(op, expr, compiler):
    # source
    relation = translate(op.table, compiler)

    # projection
    relation = compiler.ir.Relation(
        impl_type=compiler.ir.RelationImpl.Project,
        impl=compiler.ir.Project(
            base=compiler.ir.RelBase(
                output_mapping=compiler.ir.PassThrough(),
                output_mapping_type=compiler.ir.Emit.PassThrough,
            ),
            rel=relation,
            expressions=[translate(expr, compiler) for expr in op.selections],
        ),
    )

    # filter
    if op.predicates:
        predicate = functools.reduce(operator.and_, op.predicates)
        relation = compiler.ir.Relation(
            impl_type=compiler.ir.RelationImpl.Filter,
            impl=compiler.ir.Filter(
                base=compiler.ir.RelBase(
                    output_mapping=compiler.ir.PassThrough(),
                    output_mapping_type=compiler.ir.Emit.PassThrough,
                ),
                rel=relation,
                predicate=translate(predicate, compiler),
            ),
        )

    # order by
    if op.sort_keys:
        relation = compiler.ir.Relation(
            impl_type=compiler.ir.RelationImpl.OrderBy,
            impl=compiler.ir.OrderBy(
                base=compiler.ir.RelBase(
                    output_mapping=compiler.ir.PassThrough(),
                    output_mapping_type=compiler.ir.Emit.PassThrough,
                ),
                rel=relation,
                keys=[translate(expr, compiler) for expr in op.sort_keys],
            ),
        )

    return compiler.ir.Plan(sinks=[relation])


@translate.register(ops.Aggregation)
def aggregation(op, expr, compiler):
    relation = translate(op.table, compiler)

    if op.predicates:
        predicate = functools.reduce(operator.and_, op.predicates)
        relation = compiler.ir.Relation(
            impl_type=compiler.ir.RelationImpl.Filter,
            impl=compiler.ir.Filter(
                base=compiler.ir.RelBase(
                    output_mapping=compiler.ir.PassThrough(),
                    output_mapping_type=compiler.ir.Emit.PassThrough,
                ),
                rel=relation,
                predicate=translate(predicate, compiler),
            ),
        )

    relation = compiler.ir.Relation(
        impl_type=compiler.ir.RelationImpl.Aggregate,
        impl=compiler.ir.Aggregate(
            base=compiler.ir.RelBase(
                output_mapping=compiler.ir.PassThrough(),
                output_mapping_type=compiler.ir.Emit.PassThrough,
            ),
            rel=relation,
            measures=[translate(expr, compiler) for expr in op.metrics],
            groupings=[
                compiler.ir.Grouping(
                    keys=[translate(expr, compiler) for expr in op.by]
                )
            ],
        ),
    )

    if op.having:
        predicate = functools.reduce(operator.and_, op.having)
        relation = compiler.ir.Relation(
            impl_type=compiler.ir.RelationImpl.Filter,
            impl=compiler.ir.Filter(
                base=compiler.ir.RelBase(
                    output_mapping=compiler.ir.PassThrough(),
                    output_mapping_type=compiler.ir.Emit.PassThrough,
                ),
                rel=relation,
                predicate=translate(predicate, compiler),
            ),
        )

    return compiler.ir.Plan(sinks=[relation])
