SELECT `string_col`, count(1) AS `count`
FROM functional_alltypes
GROUP BY 1
HAVING max(`double_col`) = 1
