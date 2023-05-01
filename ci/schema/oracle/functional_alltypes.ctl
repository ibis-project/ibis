options (SKIP=1)
load data
  infile '/opt/oracle/csv/functional_alltypes.csv'
  into table "functional_alltypes"
  fields terminated by "," optionally enclosed by '"'
  TRAILING NULLCOLS
  ( "id",
  "bool_col",
  "tinyint_col",
  "smallint_col",
  "int_col",
  "bigint_col",
  "float_col",
  "double_col",
  "date_string_col",
  "string_col",
  "timestamp_col" "to_timestamp(:\"timestamp_col\", 'YYYY-MM-DD HH24:MI:SS.FF')",
  "year",
  "month" )
