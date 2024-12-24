SELECT i_item_id,
       ca_country,
       ca_state,
       ca_county,
       avg(cast(cs_quantity AS decimal(12, 2)))      agg1,
       avg(cast(cs_list_price AS decimal(12, 2)))    agg2,
       avg(cast(cs_coupon_amt AS decimal(12, 2)))    agg3,
       avg(cast(cs_sales_price AS decimal(12, 2)))   agg4,
       avg(cast(cs_net_profit AS decimal(12, 2)))    agg5,
       avg(cast(c_birth_year AS decimal(12, 2)))     agg6,
       avg(cast(cd1.cd_dep_count AS decimal(12, 2))) agg7
FROM catalog_sales
JOIN customer_demographics cd1
  ON cs_bill_cdemo_sk = cd1.cd_demo_sk
JOIN customer
  ON cs_bill_customer_sk = c_customer_sk
JOIN customer_demographics cd2
  ON c_current_cdemo_sk = cd2.cd_demo_sk
JOIN customer_address
  ON c_current_addr_sk = ca_address_sk
JOIN date_dim
  ON cs_sold_date_sk = d_date_sk
JOIN item
  ON cs_item_sk = i_item_sk
WHERE cd1.cd_gender = 'F'
  AND cd1.cd_education_status = 'Unknown'
  AND c_birth_month IN (1,
                        6,
                        8,
                        9,
                        12,
                        2)
  AND d_year = 1998
  AND ca_state IN ('MS',
                   'IN',
                   'ND',
                   'OK',
                   'NM',
                   'VA',
                   'MS')
GROUP BY ROLLUP (i_item_id,
    ca_country,
    ca_state,
    ca_county)
ORDER BY ca_country NULLS FIRST,
    ca_state NULLS FIRST,
    ca_county NULLS FIRST,
    i_item_id NULLS FIRST
LIMIT 100;
