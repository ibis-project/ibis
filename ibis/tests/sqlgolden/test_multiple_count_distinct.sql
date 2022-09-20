SELECT `string_col`, count(DISTINCT `int_col`) AS `int_card`,
       count(DISTINCT `smallint_col`) AS `smallint_card`
FROM functional_alltypes
GROUP BY 1
