CREATE OR REPLACE TABLE `testing.functional_alltypes_parted`
(
  index INT64,
  Unnamed_0 INT64,
  id INT64,
  bool_col BOOL,
  tinyint_col INT64,
  smallint_col INT64,
  int_col INT64,
  bigint_col INT64,
  float_col FLOAT64,
  double_col FLOAT64,
  date_string_col STRING,
  string_col STRING,
  timestamp_col TIMESTAMP,
  year INT64,
  month INT64
)
PARTITION BY DATE(_PARTITIONTIME)
OPTIONS (
  require_partition_filter=false
);

CREATE OR REPLACE TABLE `testing.functional_alltypes`
(
  index INT64,
  Unnamed_0 INT64,
  id INT64,
  bool_col BOOL,
  tinyint_col INT64,
  smallint_col INT64,
  int_col INT64,
  bigint_col INT64,
  float_col FLOAT64,
  double_col FLOAT64,
  date_string_col STRING,
  string_col STRING,
  timestamp_col TIMESTAMP,
  year INT64,
  month INT64
);