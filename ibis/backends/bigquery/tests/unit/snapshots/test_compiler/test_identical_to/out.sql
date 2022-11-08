SELECT *
FROM functional_alltypes
WHERE (`string_col` IS NOT DISTINCT FROM 'a') AND
      (`date_string_col` IS NOT DISTINCT FROM 'b')