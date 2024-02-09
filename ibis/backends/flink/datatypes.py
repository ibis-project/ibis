from __future__ import annotations

from typing import TYPE_CHECKING

from pyflink.table.types import (
    ArrayType,
    DataType,
    DataTypes,
    MapType,
    RowType,
    _from_java_data_type,
)

import ibis.expr.datatypes as dt
import ibis.expr.schema as sch
from ibis.formats import SchemaMapper, TypeMapper

if TYPE_CHECKING:
    from pyflink.table import TableSchema


class FlinkRowSchema(SchemaMapper):
    @classmethod
    def from_ibis(cls, schema: sch.Schema | None) -> list[RowType]:
        if schema is None:
            return None

        return DataTypes.ROW(
            [
                DataTypes.FIELD(k, FlinkType.from_ibis(v))
                for k, v in schema.fields.items()
            ]
        )


class FlinkType(TypeMapper):
    @classmethod
    def to_ibis(cls, typ: DataType) -> dt.DataType:
        """Convert a flink type to an ibis type."""
        nullable = typ.nullable
        if typ == DataTypes.STRING():
            return dt.String(nullable=nullable)
        elif typ == DataTypes.BOOLEAN():
            return dt.Boolean(nullable=nullable)
        elif typ == DataTypes.BYTES():
            return dt.Binary(nullable=nullable)
        elif typ == DataTypes.TINYINT():
            return dt.Int8(nullable=nullable)
        elif typ == DataTypes.SMALLINT():
            return dt.Int16(nullable=nullable)
        elif typ == DataTypes.INT():
            return dt.Int32(nullable=nullable)
        elif typ == DataTypes.BIGINT():
            return dt.Int64(nullable=nullable)
        elif typ == DataTypes.FLOAT():
            return dt.Float32(nullable=nullable)
        elif typ == DataTypes.DOUBLE():
            return dt.Float64(nullable=nullable)
        elif typ == DataTypes.DATE():
            return dt.Date(nullable=nullable)
        elif typ == DataTypes.TIME():
            return dt.Time(nullable=nullable)
        elif typ == DataTypes.TIMESTAMP():
            return dt.Timestamp(scale=typ.precision, nullable=nullable)
        elif isinstance(typ, ArrayType):
            return dt.Array(value_type=cls.to_ibis(typ.element_type), nullable=nullable)
        elif isinstance(typ, MapType):
            return dt.Map(
                key_type=cls.to_ibis(typ.key_type),
                value_type=cls.to_ibis(typ.value_type),
                nullable=nullable,
            )
        elif isinstance(typ, RowType):
            return dt.Struct(
                {field.name: cls.to_ibis(field.data_type) for field in typ.fields},
                nullable=nullable,
            )
        else:
            return super().to_ibis(typ, nullable=nullable)

    @classmethod
    def from_ibis(cls, dtype: dt.DataType) -> DataType:
        """Convert an ibis type to a flink type."""
        nullable = dtype.nullable
        if dtype.is_string():
            return DataTypes.STRING(nullable=nullable)
        elif dtype.is_boolean():
            return DataTypes.BOOLEAN(nullable=nullable)
        elif dtype.is_binary():
            return DataTypes.BYTES(nullable=nullable)
        elif dtype.is_int8():
            return DataTypes.TINYINT(nullable=nullable)
        elif dtype.is_int16():
            return DataTypes.SMALLINT(nullable=nullable)
        elif dtype.is_int32():
            return DataTypes.INT(nullable=nullable)
        elif dtype.is_int64():
            return DataTypes.BIGINT(nullable=nullable)
        elif dtype.is_uint8():
            return DataTypes.TINYINT(nullable=nullable)
        elif dtype.is_uint16():
            return DataTypes.SMALLINT(nullable=nullable)
        elif dtype.is_uint32():
            return DataTypes.INT(nullable=nullable)
        elif dtype.is_uint64():
            return DataTypes.BIGINT(nullable=nullable)
        elif dtype.is_float16():
            return DataTypes.FLOAT(nullable=nullable)
        elif dtype.is_float32():
            return DataTypes.FLOAT(nullable=nullable)
        elif dtype.is_float64():
            return DataTypes.DOUBLE(nullable=nullable)
        elif dtype.is_date():
            return DataTypes.DATE(nullable=nullable)
        elif dtype.is_time():
            return DataTypes.TIME(nullable=nullable)
        elif dtype.is_timestamp():
            # Note (mehmet): If `precision` is None, set it to 6.
            # This is because `DataTypes.TIMESTAMP` throws TypeError
            # if `precision` is None, and assumes `precision = 6`
            # if it is not provided.
            return DataTypes.TIMESTAMP(
                precision=dtype.scale if dtype.scale is not None else 6,
                nullable=nullable,
            )
        elif dtype.is_array():
            return DataTypes.ARRAY(cls.from_ibis(dtype.value_type), nullable=nullable)
        elif dtype.is_map():
            return DataTypes.MAP(
                key_type=cls.from_ibis(dtype.key_type),
                value_type=cls.from_ibis(dtype.key_type),
                nullable=nullable,
            )
        elif dtype.is_struct():
            return DataTypes.ROW(
                [
                    DataTypes.FIELD(name, data_type=cls.from_ibis(typ))
                    for name, typ in dtype.items()
                ],
                nullable=nullable,
            )
        else:
            return super().from_ibis(dtype)

    @classmethod
    def to_string(cls, dtype: dt.DataType) -> str:
        return cls.from_ibis(dtype).type_name()


def get_field_data_types(pyflink_schema: TableSchema) -> list[DataType]:
    """Returns all field data types in `pyflink_schema` as a list.

    This is a re-implementation of `get_field_data_types()` available for PyFlink
    schemas. PyFlink's implementation currently supports only `precision = 3` for
    `TimestampType` (for some reason that we could not figure out -- might be just
    a bug). The lack of precision support led to an error due to unmatched schemas
    for batches and the file to write in `to_csv()` and `to_parquet()`.

    :return: A list of all field data types.
    """
    from pyflink.java_gateway import get_gateway
    from pyflink.util.java_utils import is_instance_of

    gateway = get_gateway()

    data_type_list = []
    for j_data_type in pyflink_schema._j_table_schema.getFieldDataTypes():
        if not is_instance_of(j_data_type, gateway.jvm.AtomicDataType):
            data_type = _from_java_data_type(j_data_type)

        else:
            logical_type = j_data_type.getLogicalType()
            if is_instance_of(logical_type, gateway.jvm.TimestampType):
                data_type = DataTypes.TIMESTAMP(
                    precision=logical_type.getPrecision(),
                    nullable=logical_type.isNullable(),
                )

            else:
                data_type = _from_java_data_type(j_data_type)

        data_type_list.append(data_type)

    return data_type_list
