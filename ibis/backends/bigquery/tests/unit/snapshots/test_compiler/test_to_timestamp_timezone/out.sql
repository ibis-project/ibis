SELECT PARSE_TIMESTAMP('%F %Z', CONCAT(`date_string_col`, ' America/New_York')) AS `tmp`
FROM functional_alltypes