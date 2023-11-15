CREATE OR REPLACE TABLE diamonds (
    "carat" FLOAT,
    "cut" TEXT,
    "color" TEXT,
    "clarity" TEXT,
    "depth" FLOAT,
    "table" FLOAT,
    "price" BIGINT,
    "x" FLOAT,
    "y" FLOAT,
    "z" FLOAT
);

CREATE OR REPLACE TABLE astronauts (
    "id" BIGINT,
    "number" BIGINT,
    "nationwide_number" BIGINT,
    "name" TEXT,
    "original_name" TEXT,
    "sex" TEXT,
    "year_of_birth" BIGINT,
    "nationality" TEXT,
    "military_civilian" TEXT,
    "selection" TEXT,
    "year_of_selection" BIGINT,
    "mission_number" BIGINT,
    "total_number_of_missions" BIGINT,
    "occupation" TEXT,
    "year_of_mission" BIGINT,
    "mission_title" TEXT,
    "ascend_shuttle" TEXT,
    "in_orbit" TEXT,
    "descend_shuttle" TEXT,
    "hours_mission" FLOAT,
    "total_hrs_sum" FLOAT,
    "field21" BIGINT,
    "eva_hrs_mission" FLOAT,
    "total_eva_hrs" FLOAT
);

CREATE OR REPLACE TABLE batting (
    "playerID" TEXT,
    "yearID" BIGINT,
    "stint" BIGINT,
    "teamID" TEXT,
    "lgID" TEXT,
    "G" BIGINT,
    "AB" BIGINT,
    "R" BIGINT,
    "H" BIGINT,
    "X2B" BIGINT,
    "X3B" BIGINT,
    "HR" BIGINT,
    "RBI" BIGINT,
    "SB" BIGINT,
    "CS" BIGINT,
    "BB" BIGINT,
    "SO" BIGINT,
    "IBB" BIGINT,
    "HBP" BIGINT,
    "SH" BIGINT,
    "SF" BIGINT,
    "GIDP" BIGINT
);

CREATE OR REPLACE TABLE awards_players (
    "playerID" TEXT,
    "awardID" TEXT,
    "yearID" BIGINT,
    "lgID" TEXT,
    "tie" TEXT,
    "notes" TEXT
);

CREATE OR REPLACE TABLE functional_alltypes (
    "id" INTEGER,
    "bool_col" BOOLEAN,
    "tinyint_col" SMALLINT,
    "smallint_col" SMALLINT,
    "int_col" INTEGER,
    "bigint_col" BIGINT,
    "float_col" REAL,
    "double_col" DOUBLE PRECISION,
    "date_string_col" TEXT,
    "string_col" TEXT,
    "timestamp_col" TIMESTAMP WITHOUT TIME ZONE,
    "year" INTEGER,
    "month" INTEGER
);

CREATE OR REPLACE TABLE array_types (
    "x" ARRAY,
    "y" ARRAY,
    "z" ARRAY,
    "grouper" TEXT,
    "scalar_column" DOUBLE PRECISION,
    "multi_dim" ARRAY
);

INSERT INTO array_types ("x", "y", "z", "grouper", "scalar_column", "multi_dim")
    SELECT [1, 2, 3], ['a', 'b', 'c'], [1.0, 2.0, 3.0], 'a', 1.0, [[], [1, 2, 3], NULL] UNION
    SELECT [4, 5], ['d', 'e'], [4.0, 5.0], 'a', 2.0, [] UNION
    SELECT [6, NULL], ['f', NULL], [6.0, NULL], 'a', 3.0, [NULL, [], NULL] UNION
    SELECT [NULL, 1, NULL], [NULL, 'a', NULL], [], 'b', 4.0, [[1], [2], [], [3, 4, 5]] UNION
    SELECT [2, NULL, 3], ['b', NULL, 'c'], NULL, 'b', 5.0, NULL UNION
    SELECT [4, NULL, NULL, 5], ['d', NULL, NULL, 'e'], [4.0, NULL, NULL, 5.0], 'c', 6.0, [[1, 2, 3]];

CREATE OR REPLACE TABLE map ("idx" BIGINT, "kv" OBJECT);

INSERT INTO map ("idx", "kv")
    SELECT 1, object_construct('a', 1, 'b', 2, 'c', 3) UNION
    SELECT 2, object_construct('d', 4, 'e', 5, 'f', 6);


CREATE OR REPLACE TABLE struct ("abc" OBJECT);

INSERT INTO struct ("abc")
    SELECT {'a': 1.0, 'b': 'banana', 'c': 2} UNION
    SELECT {'a': 2.0, 'b': 'apple', 'c': 3} UNION
    SELECT {'a': 3.0, 'b': 'orange', 'c': 4} UNION
    SELECT {'a': NULL, 'b': 'banana', 'c': 2} UNION
    SELECT {'a': 2.0, 'b': NULL, 'c': 3} UNION
    SELECT NULL UNION
    SELECT {'a': 3.0, 'b': 'orange', 'c': NULL};

CREATE OR REPLACE TABLE json_t ("js" VARIANT);

INSERT INTO json_t ("js")
    SELECT parse_json('{"a": [1,2,3,4], "b": 1}') UNION
    SELECT parse_json('{"a":null,"b":2}') UNION
    SELECT parse_json('{"a":"foo", "c":null}') UNION
    SELECT parse_json('null') UNION
    SELECT parse_json('[42,47,55]') UNION
    SELECT parse_json('[]');

CREATE OR REPLACE TABLE win ("g" TEXT, "x" BIGINT NOT NULL, "y" BIGINT);
INSERT INTO win VALUES
    ('a', 0, 3),
    ('a', 1, 2),
    ('a', 2, 0),
    ('a', 3, 1),
    ('a', 4, 1);
