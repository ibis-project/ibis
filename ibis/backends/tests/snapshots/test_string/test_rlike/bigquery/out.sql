SELECT
  t0.*
FROM `ibis-gbq.ibis_gbq_testing.functional_alltypes` AS t0
WHERE
  REGEXP_CONTAINS(t0.`string_col`, '0')