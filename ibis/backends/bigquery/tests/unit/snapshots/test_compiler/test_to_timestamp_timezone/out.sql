SELECT
  PARSE_TIMESTAMP('%F %Z', CONCAT(t0.`date_string_col`, ' America/New_York')) AS `StringToTimestamp_StringConcat_ '%F %Z'`
FROM functional_alltypes AS t0