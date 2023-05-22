SELECT t0.`string_col`, count(DISTINCT t0.`int_col`) AS `int_card`,
       count(DISTINCT t0.`smallint_col`) AS `smallint_card`
FROM functional_alltypes t0
GROUP BY 1