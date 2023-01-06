SELECT PARSE_TIMESTAMP('%F %Z', CONCAT(t0.`date_string_col`, ' America/New_York')) AS `StringToTimestamp_StringConcat_F_Z`
FROM functional_alltypes t0