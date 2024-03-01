from __future__ import annotations

try:
    from duckdb import ConversionException as DuckDBConversionException
    from duckdb import InvalidInputException as DuckDBInvalidInputException
    from duckdb import NotImplementedException as DuckDBNotImplementedException
    from duckdb import ParserException as DuckDBParserException
except ImportError:
    DuckDBConversionException = DuckDBInvalidInputException = DuckDBParserException = (
        DuckDBNotImplementedException
    ) = None

try:
    from clickhouse_connect.driver.exceptions import (
        DatabaseError as ClickHouseDatabaseError,
    )
    from clickhouse_connect.driver.exceptions import (
        InternalError as ClickHouseInternalError,
    )
    from clickhouse_connect.driver.exceptions import (
        OperationalError as ClickHouseOperationalError,
    )
except ImportError:
    ClickHouseDatabaseError = ClickHouseInternalError = ClickHouseOperationalError = (
        None
    )


try:
    from pyexasol.exceptions import ExaQueryError
except ImportError:
    ExaQueryError = None

try:
    from pyspark.sql.utils import AnalysisException as PySparkAnalysisException
    from pyspark.sql.utils import (
        IllegalArgumentException as PySparkIllegalArgumentException,
    )
    from pyspark.sql.utils import ParseException as PySparkParseException
    from pyspark.sql.utils import PythonException as PySparkPythonException
except ImportError:
    PySparkAnalysisException = PySparkIllegalArgumentException = (
        PySparkParseException
    ) = PySparkPythonException = None

try:
    # PySpark 3.5.0
    from pyspark.errors.exceptions.captured import (
        ArithmeticException as PySparkArithmeticException,
    )
except ImportError:
    PySparkArithmeticException = None

try:
    from google.api_core.exceptions import BadRequest as GoogleBadRequest
except ImportError:
    GoogleBadRequest = None

try:
    from polars import ComputeError as PolarsComputeError
    from polars import PanicException as PolarsPanicException
    from polars.exceptions import InvalidOperationError as PolarsInvalidOperationError
except ImportError:
    PolarsComputeError = PolarsPanicException = PolarsInvalidOperationError = None

try:
    from pyarrow import ArrowInvalid, ArrowNotImplementedError
except ImportError:
    ArrowInvalid = ArrowNotImplementedError = None

try:
    from impala.error import HiveServer2Error as ImpalaHiveServer2Error
    from impala.error import OperationalError as ImpalaOperationalError
except ImportError:
    ImpalaHiveServer2Error = ImpalaOperationalError = None

try:
    from py4j.protocol import Py4JError, Py4JJavaError
except ImportError:
    Py4JJavaError = Py4JError = None

try:
    from deltalake import PyDeltaTableError
except ImportError:
    PyDeltaTableError = None

try:
    from snowflake.connector.errors import ProgrammingError as SnowflakeProgrammingError
except ImportError:
    SnowflakeProgrammingError = None

try:
    from trino.exceptions import TrinoUserError
except ImportError:
    TrinoUserError = None

try:
    from psycopg2.errors import DivisionByZero as PsycoPg2DivisionByZero
    from psycopg2.errors import IndeterminateDatatype as PsycoPg2IndeterminateDatatype
    from psycopg2.errors import InternalError_ as PsycoPg2InternalError
    from psycopg2.errors import (
        InvalidTextRepresentation as PsycoPg2InvalidTextRepresentation,
    )
    from psycopg2.errors import OperationalError as PsycoPg2OperationalError
    from psycopg2.errors import ProgrammingError as PsycoPg2ProgrammingError
    from psycopg2.errors import SyntaxError as PsycoPg2SyntaxError
    from psycopg2.errors import UndefinedObject as PsycoPg2UndefinedObject
except ImportError:
    PsycoPg2SyntaxError = PsycoPg2IndeterminateDatatype = (
        PsycoPg2InvalidTextRepresentation
    ) = PsycoPg2DivisionByZero = PsycoPg2InternalError = PsycoPg2ProgrammingError = (
        PsycoPg2OperationalError
    ) = PsycoPg2UndefinedObject = None

try:
    from pymysql.err import NotSupportedError as MySQLNotSupportedError
    from pymysql.err import OperationalError as MySQLOperationalError
    from pymysql.err import ProgrammingError as MySQLProgrammingError
except ImportError:
    MySQLNotSupportedError = MySQLProgrammingError = MySQLOperationalError = None

try:
    from pydruid.db.exceptions import ProgrammingError as PyDruidProgrammingError
except ImportError:
    PyDruidProgrammingError = None

try:
    from oracledb.exceptions import DatabaseError as OracleDatabaseError
except ImportError:
    OracleDatabaseError = None

try:
    from pyodbc import DataError as PyODBCDataError
    from pyodbc import ProgrammingError as PyODBCProgrammingError
except ImportError:
    PyODBCProgrammingError = PyODBCDataError = None
