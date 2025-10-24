"""Beam SQL dialect for SQLGlot."""

from __future__ import annotations

import sqlglot as sg
from sqlglot.dialects import Dialect


class Beam(Dialect):
    """Beam SQL dialect."""
    
    class Tokenizer(sg.Tokenizer):
        """Beam SQL tokenizer."""
        
        KEYWORDS = {
            **sg.Tokenizer.KEYWORDS,
            "ARRAY": "ARRAY",
            "MAP": "MAP",
            "ROW": "ROW",
            "STRUCT": "STRUCT",
            "WATERMARK": "WATERMARK",
            "PARTITIONED": "PARTITIONED",
            "BY": "BY",
            "WITH": "WITH",
            "PROPERTIES": "PROPERTIES",
            "CONNECTOR": "CONNECTOR",
            "FORMAT": "FORMAT",
            "PATH": "PATH",
            "SCHEMA": "SCHEMA",
            "PRIMARY": "PRIMARY",
            "KEY": "KEY",
            "NOT": "NOT",
            "ENFORCED": "ENFORCED",
        }

    class Parser(sg.Parser):
        """Beam SQL parser."""
        
        FUNCTIONS = {
            **sg.Parser.FUNCTIONS,
            "array_agg": "ARRAY_AGG",
            "array_agg_distinct": "ARRAY_AGG_DISTINCT",
            "array_concat": "ARRAY_CONCAT",
            "array_contains": "ARRAY_CONTAINS",
            "array_get": "ARRAY_GET",
            "array_repeat": "ARRAY_REPEAT",
            "array_slice": "ARRAY_SLICE",
            "array_sort": "ARRAY_SORT",
            "array_union": "ARRAY_UNION",
            "array_distinct": "ARRAY_DISTINCT",
            "cardinality": "CARDINALITY",
            "array_position": "ARRAY_POSITION",
            "array_remove": "ARRAY_REMOVE",
            "map_keys": "MAP_KEYS",
            "map_values": "MAP_VALUES",
            "map_get": "MAP_GET",
            "map_contains": "MAP_CONTAINS",
            "map_merge": "MAP_MERGE",
            "map_concat": "MAP_CONCAT",
            "map_from_arrays": "MAP_FROM_ARRAYS",
            "map_from_entries": "MAP_FROM_ENTRIES",
            "map_entries": "MAP_ENTRIES",
            "approx_count_distinct": "APPROX_COUNT_DISTINCT",
            "dayofyear": "DAYOFYEAR",
            "power": "POWER",
            "regexp": "REGEXP",
            "right": "RIGHT",
            "char_length": "CHAR_LENGTH",
            "to_date": "TO_DATE",
            "to_timestamp": "TO_TIMESTAMP",
            "typeof": "TYPEOF",
        }

    class Generator(sg.Generator):
        """Beam SQL generator."""
        
        TYPE_MAPPING = {
            **sg.Generator.TYPE_MAPPING,
            sg.DataType.Type.TINYINT: "TINYINT",
            sg.DataType.Type.SMALLINT: "SMALLINT",
            sg.DataType.Type.INT: "INTEGER",
            sg.DataType.Type.BIGINT: "BIGINT",
            sg.DataType.Type.FLOAT: "FLOAT",
            sg.DataType.Type.DOUBLE: "DOUBLE",
            sg.DataType.Type.BOOLEAN: "BOOLEAN",
            sg.DataType.Type.STRING: "VARCHAR",
            sg.DataType.Type.CHAR: "CHAR",
            sg.DataType.Type.VARCHAR: "VARCHAR",
            sg.DataType.Type.DATE: "DATE",
            sg.DataType.Type.TIME: "TIME",
            sg.DataType.Type.TIMESTAMP: "TIMESTAMP",
            sg.DataType.Type.ARRAY: "ARRAY",
            sg.DataType.Type.MAP: "MAP",
            sg.DataType.Type.STRUCT: "ROW",
        }

        def array_sql(self, expression):
            """Generate ARRAY type SQL."""
            return f"ARRAY<{self.sql(expression, 'expressions')[0]}>"

        def map_sql(self, expression):
            """Generate MAP type SQL."""
            key_type = self.sql(expression, 'expressions')[0]
            value_type = self.sql(expression, 'expressions')[1]
            return f"MAP<{key_type}, {value_type}>"

        def struct_sql(self, expression):
            """Generate ROW type SQL."""
            fields = []
            for field in expression.expressions:
                field_name = field.alias or field.this
                field_type = self.sql(field, 'expressions')[0]
                fields.append(f"{field_name} {field_type}")
            return f"ROW({', '.join(fields)})"

        def watermark_sql(self, expression):
            """Generate WATERMARK SQL."""
            time_col = expression.this
            strategy = expression.expressions[0] if expression.expressions else None
            if strategy:
                return f"WATERMARK FOR {time_col} AS {strategy}"
            return f"WATERMARK FOR {time_col}"

        def partitioned_sql(self, expression):
            """Generate PARTITIONED BY SQL."""
            columns = self.sql(expression, 'expressions')
            return f"PARTITIONED BY ({', '.join(columns)})"

        def properties_sql(self, expression):
            """Generate WITH PROPERTIES SQL."""
            props = []
            for prop in expression.expressions:
                key = prop.this
                value = prop.expressions[0] if prop.expressions else None
                if value:
                    props.append(f"'{key}'='{value}'")
                else:
                    props.append(f"'{key}'")
            return f"WITH ({', '.join(props)})"

        def primary_key_sql(self, expression):
            """Generate PRIMARY KEY SQL."""
            columns = self.sql(expression, 'expressions')
            return f"PRIMARY KEY ({', '.join(columns)})"

        def connector_sql(self, expression):
            """Generate CONNECTOR SQL."""
            return f"CONNECTOR = '{expression.this}'"

        def format_sql(self, expression):
            """Generate FORMAT SQL."""
            return f"FORMAT = '{expression.this}'"

        def path_sql(self, expression):
            """Generate PATH SQL."""
            return f"PATH = '{expression.this}'"

        def schema_sql(self, expression):
            """Generate SCHEMA SQL."""
            return f"SCHEMA = '{expression.this}'"
