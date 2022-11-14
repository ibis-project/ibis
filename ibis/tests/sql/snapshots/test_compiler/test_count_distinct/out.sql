SELECT `string_col`, count(DISTINCT `int_col`) AS `nunique`
FROM functional_alltypes
WHERE `bigint_col` > 0
GROUP BY 1