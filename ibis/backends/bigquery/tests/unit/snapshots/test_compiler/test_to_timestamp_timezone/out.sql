SELECT
  PARSE_TIMESTAMP('%F %Z', CONCAT(`t0`.`date_string_col`, ' America/New_York'), 'UTC') AS `StringToTimestamp_StringConcat_date_string_col_' America_New_York'_'%F %Z'`
FROM `functional_alltypes` AS `t0`