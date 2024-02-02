SELECT
  parse_timestamp('%F %Z', CONCAT(`t0`.`date_string_col`, ' America/New_York'), 'UTC') AS `StringToTimestamp_StringConcat_ '%F %Z'`
FROM `functional_alltypes` AS `t0`