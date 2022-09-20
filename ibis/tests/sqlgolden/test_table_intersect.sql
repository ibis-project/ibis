SELECT `string_col` AS `key`, CAST(`float_col` AS double) AS `value`
FROM functional_alltypes
WHERE `int_col` > 0
INTERSECT
SELECT `string_col` AS `key`, `double_col` AS `value`
FROM functional_alltypes
WHERE `int_col` <= 0
