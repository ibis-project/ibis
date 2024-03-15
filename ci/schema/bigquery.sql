CREATE SCHEMA IF NOT EXISTS ibis_gbq_testing;

CREATE OR REPLACE TABLE ibis_gbq_testing.struct (
    abc STRUCT<a FLOAT64, b STRING, c INT64>
);

INSERT INTO ibis_gbq_testing.struct VALUES
    (STRUCT(1.0, 'banana', 2)),
    (STRUCT(2.0, 'apple', 3)),
    (STRUCT(3.0, 'orange', 4)),
    (STRUCT(NULL, 'banana', 2)),
    (STRUCT(2.0, NULL, 3)),
    (NULL),
    (STRUCT(3.0, 'orange', NULL));

CREATE OR REPLACE TABLE ibis_gbq_testing.array_types (
    x ARRAY<INT64>,
    y ARRAY<STRING>,
    z ARRAY<FLOAT64>,
    grouper STRING,
    scalar_column FLOAT64,
);

INSERT INTO ibis_gbq_testing.array_types VALUES
    ([1, 2, 3], ['a', 'b', 'c'], [1.0, 2.0, 3.0], 'a', 1.0),
    ([4, 5], ['d', 'e'], [4.0, 5.0], 'a', 2.0),
    ([6], ['f'], [6.0], 'a', 3.0),
    ([1], ['a'], [], 'b', 4.0),
    ([2, 3], ['b', 'c'], NULL, 'b', 5.0),
    ([4, 5], ['d', 'e'], [4.0, 5.0], 'c', 6.0);

CREATE OR REPLACE TABLE ibis_gbq_testing.win (
    g STRING,
    x INT64,
    y INT64
);

INSERT INTO ibis_gbq_testing.win VALUES
    ('a', 0, 3),
    ('a', 1, 2),
    ('a', 2, 0),
    ('a', 3, 1),
    ('a', 4, 1);

CREATE OR REPLACE TABLE ibis_gbq_testing.topk (
    x INT64
);

INSERT INTO ibis_gbq_testing.topk VALUES (1), (1), (NULL);

CREATE OR REPLACE TABLE ibis_gbq_testing.numeric_table (
    string_col STRING,
    numeric_col NUMERIC
);

INSERT INTO ibis_gbq_testing.numeric_table VALUES
    ('1st value', 0.999999999),
    ('2nd value', 0.000000002);

CREATE OR REPLACE TABLE ibis_gbq_testing.json_t (
    js JSON
);

INSERT INTO ibis_gbq_testing.json_t VALUES
    (JSON '{"a": [1,2,3,4], "b": 1}'),
    (JSON '{"a":null,"b":2}'),
    (JSON '{"a":"foo", "c":null}'),
    (JSON 'null'),
    (JSON '[42,47,55]'),
    (JSON '[]');


LOAD DATA OVERWRITE ibis_gbq_testing.functional_alltypes (
    id INT64,
    bool_col BOOLEAN,
    tinyint_col INT64,
    smallint_col INT64,
    int_col INT64,
    bigint_col INT64,
    float_col FLOAT64,
    double_col FLOAT64,
    date_string_col STRING,
    string_col STRING,
    timestamp_col DATETIME,
    year INT64,
    month INT64
)
FROM FILES (
    format = 'PARQUET',
    uris = ['gs://ibis-ci-data/functional_alltypes.parquet']
);

LOAD DATA OVERWRITE ibis_gbq_testing.awards_players
FROM FILES (
    format = 'PARQUET',
    uris = ['gs://ibis-ci-data/awards_players.parquet']
);

LOAD DATA OVERWRITE ibis_gbq_testing.batting
FROM FILES (
    format = 'PARQUET',
    uris = ['gs://ibis-ci-data/batting.parquet']
);

LOAD DATA OVERWRITE ibis_gbq_testing.diamonds
FROM FILES (
    format = 'PARQUET',
    uris = ['gs://ibis-ci-data/diamonds.parquet']
);

LOAD DATA OVERWRITE ibis_gbq_testing.astronauts
FROM FILES (
    format = 'PARQUET',
    uris = ['gs://ibis-ci-data/astronauts.parquet']
);

LOAD DATA OVERWRITE ibis_gbq_testing.functional_alltypes_parted (
    id INT64,
    bool_col BOOLEAN,
    tinyint_col INT64,
    smallint_col INT64,
    int_col INT64,
    bigint_col INT64,
    float_col FLOAT64,
    double_col FLOAT64,
    date_string_col STRING,
    string_col STRING,
    timestamp_col DATETIME,
    year INT64,
    month INT64
)
PARTITION BY _PARTITIONDATE
FROM FILES (
    format = 'PARQUET',
    uris = ['gs://ibis-ci-data/functional_alltypes.parquet']
);

CREATE OR REPLACE TABLE ibis_gbq_testing.timestamp_column_parted (
    my_timestamp_parted_col TIMESTAMP,
    string_col STRING,
    int_col INT64
)
PARTITION BY DATE(my_timestamp_parted_col);

CREATE OR REPLACE TABLE ibis_gbq_testing.date_column_parted (
    my_date_parted_col DATE,
    string_col STRING,
    int_col INT64
)
PARTITION BY my_date_parted_col;

CREATE OR REPLACE TABLE ibis_gbq_testing.struct_table (
    array_of_structs_col ARRAY<STRUCT<int_field INTEGER, string_field STRING>>,
    nested_struct_col STRUCT<sub_struct STRUCT<timestamp_col TIMESTAMP>>,
    struct_col STRUCT<string_field STRING>
);

INSERT INTO ibis_gbq_testing.struct_table VALUES
    ([(12345, 'abcdefg'), (NULL, NULL)],
     STRUCT(STRUCT(NULL)),
     STRUCT(NULL)),
    ([(12345, 'abcdefg'), (NULL, 'hijklmnop')],
     STRUCT(STRUCT('2017-10-20 16:37:50.000000')),
     STRUCT('a'));
