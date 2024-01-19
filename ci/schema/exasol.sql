DROP SCHEMA IF EXISTS EXASOL CASCADE;
CREATE SCHEMA EXASOL;

CREATE OR REPLACE TABLE EXASOL."diamonds"
(
    "carat"   DOUBLE,
    "cut"     VARCHAR(256),
    "color"   VARCHAR(256),
    "clarity" VARCHAR(256),
    "depth"   DOUBLE,
    "table"   DOUBLE,
    "price"   BIGINT,
    "x"       DOUBLE,
    "y"       DOUBLE,
    "z"       DOUBLE
);

CREATE OR REPLACE TABLE EXASOL."batting"
(
    "playerID" VARCHAR(256),
    "yearID"   BIGINT,
    "stint"    BIGINT,
    "teamID"   VARCHAR(256),
    "lgID"     VARCHAR(256),
    "G"        BIGINT,
    "AB"       BIGINT,
    "R"        BIGINT,
    "H"        BIGINT,
    "X2B"      BIGINT,
    "X3B"      BIGINT,
    "HR"       BIGINT,
    "RBI"      BIGINT,
    "SB"       BIGINT,
    "CS"       BIGINT,
    "BB"       BIGINT,
    "SO"       BIGINT,
    "IBB"      BIGINT,
    "HBP"      BIGINT,
    "SH"       BIGINT,
    "SF"       BIGINT,
    "GIDP"     BIGINT
);

CREATE OR REPLACE TABLE EXASOL."awards_players"
(
    "playerID" VARCHAR(256),
    "awardID"  VARCHAR(256),
    "yearID"   BIGINT,
    "lgID"     VARCHAR(256),
    "tie"      VARCHAR(256),
    "notest"   VARCHAR(256)
);

CREATE OR REPLACE TABLE EXASOL."functional_alltypes"
(
    "id"              INTEGER,
    "bool_col"        BOOLEAN,
    "tinyint_col"     SHORTINT,
    "smallint_col"    SMALLINT,
    "int_col"         INTEGER,
    "bigint_col"      BIGINT,
    "float_col"       FLOAT,
    "double_col"      DOUBLE PRECISION,
    "date_string_col" VARCHAR(256),
    "string_col"      VARCHAR(256),
    "timestamp_col"   TIMESTAMP,
    "year"            INTEGER,
    "month"           INTEGER
);


IMPORT INTO EXASOL."diamonds" FROM LOCAL CSV FILE '/data/diamonds.csv' COLUMN SEPARATOR = ',' SKIP = 1;
IMPORT INTO EXASOL."batting" FROM LOCAL CSV FILE '/data/batting.csv' COLUMN SEPARATOR = ',' SKIP = 1;
IMPORT INTO EXASOL."awards_players" FROM LOCAL CSV FILE '/data/awards_players.csv' COLUMN SEPARATOR = ',' SKIP = 1;
IMPORT INTO EXASOL."functional_alltypes" FROM LOCAL CSV FILE '/data/functional_alltypes.csv' COLUMN SEPARATOR = ',' SKIP = 1;

CREATE OR REPLACE TABLE EXASOL."win"
(
    "g" VARCHAR(1),
    "x" BIGINT,
    "y" BIGINT
);

INSERT INTO "win" VALUES
    ('a', 0, 3),
    ('a', 1, 2),
    ('a', 2, 0),
    ('a', 3, 1),
    ('a', 4, 1);
