from __future__ import annotations

try:
    from duckdb import ConversionException as DuckDBConversionException
    from duckdb import InvalidInputException as DuckDBInvalidInputException
except ImportError:
    DuckDBConversionException = DuckDBInvalidInputException = None

try:
    from clickhouse_connect.driver.exceptions import (
        DatabaseError as ClickHouseDatabaseError,
    )
    from clickhouse_connect.driver.exceptions import (
        InternalError as ClickHouseInternalError,
    )
except ImportError:
    ClickHouseDatabaseError = ClickHouseInternalError = None

try:
    from pyexasol.exceptions import ExaQueryError
except ImportError:
    ExaQueryError = None

try:
    from pyspark.sql.utils import AnalysisException as PySparkAnalysisException
    from pyspark.sql.utils import (
        IllegalArgumentException as PySparkIllegalArgumentException,
    )
    from pyspark.sql.utils import PythonException as PySparkPythonException
except ImportError:
    PySparkAnalysisException = (
        PySparkIllegalArgumentException
    ) = PySparkPythonException = None

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
