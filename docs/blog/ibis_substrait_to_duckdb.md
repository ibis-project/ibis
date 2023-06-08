# Ibis + Substrait + DuckDB

**by Gil Forsyth**

Ibis strives to provide a consistent interface for interacting with a multitude
of different analytical execution engines, most of which (but not all) speak
some dialect of SQL.

Today, Ibis accomplishes this with a lot of help from `sqlalchemy` and `sqlglot`
to handle differences in dialect, or we interact directly with available Python
bindings (for instance with the `pandas`, `datafusion`, and `polars` backends).

Ibis goes to <span class="underline">great</span> lengths to generate sane and consistent SQL for those
backends that use it. We are also interested in exploring other means of
communicating consistently with those backends.

[Substrait](https://substrait.io/) is a new cross-language serialization format for communicating (among
other things) query plans. It's still in its early days, but there is already
nascent support for Substrait in [Apache Arrow](https://arrow.apache.org/docs/dev/cpp/streaming_execution.html#substrait), [DuckDB](https://duckdb.org/docs/extensions/substrait), and [Velox](https://engineering.fb.com/2022/08/31/open-source/velox/).

Ibis supports producing Substrait plans from Ibis table expressions, with the
help of the [ibis-substrait](https://github.com/ibis-project/ibis-substrait)
library. Let's take a quick peek at how we might use it for query execution.

## Getting started

First, we can create a `conda` environment using the latest versions of
`duckdb`, `ibis`, and `ibis_substrait`.

```sh
mamba create -n ibis_substrait_duckdb ibis-framework==4.1 ibis-substrait==2.19 ipython python-duckdb parsy==2
```

Next, we'll need to choose a dataset. For this example, we'll use data from IMDB, available through their [dataset portal](https://datasets.imdbws.com/).

For convenience, I used [Ready, Set, Data!](https://github.com/saulpw/readysetdata) to grab the data in `parquet` format and then insert
it into a DuckDB database.

```python
import duckdb
con = duckdb.connect("/home/gil/imdb.ddb")
con.execute(
    "CREATE TABLE ratings AS SELECT * FROM '/home/gil/data/imdb/imdb_ratings.parquet'"
)
con.execute(
    "CREATE TABLE basics AS SELECT * FROM '/home/gil/data/imdb/imdb_basics.parquet'"
)
```

## Query Creation

For our example, we'll build up a query using Ibis but without connecting to our
execution engine (DuckDB). Once we have an Ibis table expression, we'll create a
Substrait plan, then execute that plan directly on DuckDB to get results.

To do this, all we need is some knowledge of the schema of the tables we want to
interact with. We might get these schema from a metadata store, or possibly a
coworker, or a friendly mouse.

However we arrive at it, if we know the column names and the datatypes, we can
build up a query in Ibis, so let's do that.

```python
import ibis
from ibis import _

ratings = ibis.table(
    [
        ("tconst", "str"),
        ("averageRating", "str"),
        ("numVotes", "str"),
    ],
    name="ratings",
)

basics = ibis.table(
    [
        ("tconst", "str"),
        ("titleType", "str"),
        ("primaryTitle", "str"),
        ("originalTitle", "str"),
        ("isAdult", "str"),
        ("startYear", "str"),
        ("endYear", "str"),
        ("runtimeMinutes", "str"),
        ("genres", "str"),
    ],
    name="basics",
)
```

Now that those tables are represented in Ibis, we can start creating our query.
We'll try to recreate the top-ten movies on the IMDB leaderboard. For that,
we'll need movie titles and their respective ratings.

We know that the data we have for `ratings` looks something like the following:

```python
┏━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━┓
┃ tconst    ┃ averageRating ┃ numVotes ┃
┡━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━┩
│ string    │ string        │ string   │
├───────────┼───────────────┼──────────┤
│ tt0000001 │ 5.7           │ 1919\n   │
│ tt0000002 │ 5.8           │ 260\n    │
│ tt0000003 │ 6.5           │ 1726\n   │
│ tt0000004 │ 5.6           │ 173\n    │
│ tt0000005 │ 6.2           │ 2541\n   │
└───────────┴───────────────┴──────────┘
```

Based on the column names alone, `averageRating` is almost certainly supposed to
be a `float`, and `numVotes` should be an `integer`. We can cast those so we
can make useful comparisons between ratings and vote numbers.

```python
ratings = ratings.select(
    ratings.tconst,
    avg_rating=ratings.averageRating.cast("float"),
    num_votes=ratings.numVotes.cast("int"),
)
```

The first few rows of `basics` looks like this:

```python
┏━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━┳━━━┓
┃ tconst    ┃ titleType ┃ primaryTitle           ┃ originalTitle          ┃ isAdult ┃ startYear ┃ … ┃
┡━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━╇━━━┩
│ string    │ string    │ string                 │ string                 │ string  │ string    │ … │
├───────────┼───────────┼────────────────────────┼────────────────────────┼─────────┼───────────┼───┤
│ tt0000001 │ short     │ Carmencita             │ Carmencita             │ 0       │ 1894      │ … │
│ tt0000002 │ short     │ Le clown et ses chiens │ Le clown et ses chiens │ 0       │ 1892      │ … │
│ tt0000003 │ short     │ Pauvre Pierrot         │ Pauvre Pierrot         │ 0       │ 1892      │ … │
│ tt0000004 │ short     │ Un bon bock            │ Un bon bock            │ 0       │ 1892      │ … │
│ tt0000005 │ short     │ Blacksmith Scene       │ Blacksmith Scene       │ 0       │ 1893      │ … │
└───────────┴───────────┴────────────────────────┴────────────────────────┴─────────┴───────────┴───┘
```

In the interest of keeping things family-friendly, we can filter out any adult
films. We can filter out any IMDB titles that aren't movies, then select out the
columns `tconst` and `primaryTitle`. And we'll include `startYear` just in case
it's interesting.

```python
basics = basics.filter([basics.titleType == "movie", basics.isAdult == "0"]).select(
    "tconst",
    "primaryTitle",
    "startYear",
)
```

With the data (lightly) cleaned up, we can construct our query for top films.
We want to join the two tables `ratings` and `basics`. Then we'll order them by
`avg_rating` and `num_votes`, and include an additional filter that the movie
has to have at least 200,000 votes.

```python
topfilms = (
    ratings.join(basics, "tconst")
    .order_by([_.avg_rating.desc(), _.num_votes.desc()])
    .filter(_.num_votes > 200_000)
    .limit(10)
)
```

Now that we have an Ibis table expression, it's time for Substrait to enter
the scene.

## Substrait Serialization

We're going to import `ibis_substrait` and compile the `topfilms` table
expression into a Substrait plan.

```python
from ibis_substrait.compiler.core import SubstraitCompiler

compiler = SubstraitCompiler()

plan = compiler.compile(topfilms)

# type(plan) --> <class 'substrait.ibis.plan_pb2.Plan'>
```

Substrait is built using `protobuf`. If you look at the `repr` for `plan`,
you'll see a LOOOONG JSON-ish representation of the Substrait plan. This
representation is not really meant for human eyes.

We'll serialize the Substrait plan to disk and open it up in a separate
session, or on another machine, entirely. That's one of the notions of
Substrait: plans can be serialized and shuttled around between various systems.
It's similar to Ibis in that it allows a separation of plan creation from plan
execution.

```python
with open("topfilms.proto", "wb") as f:
    f.write(plan.SerializeToString())
```

## Substrait Plan Execution

Now we can open up the serialized Substrait plan in a new session where we
execute it using DuckDB directly. One important point to note here is that our
plan refers to two tables, named `basics` and `ratings`. If those tables don't
exist in our execution engine, then this isn't going to work.

```python
import duckdb

con = duckdb.connect("/home/gil/imdb.ddb")

con.execute("PRAGMA show_tables;").fetchall()
```

<table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">

<colgroup>
<col  class="org-left" />
</colgroup>
<tbody>
<tr>
<td class="org-left">basics</td>
</tr>

<tr>
<td class="org-left">ratings</td>
</tr>
</tbody>
</table>

Luckily, they do exist! Let's install and load the DuckDB Substrait extension,
then execute the Substrait plan, and finally grab our results.

```python
con.install_extension("substrait")
con.load_extension("substrait")

with open("topfilms.proto", "rb") as f:
    plan_blob = f.read()

result = con.from_substrait(plan_blob)

result.fetchall()
```

<table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">

<colgroup>
<col  class="org-left" />

<col  class="org-right" />

<col  class="org-right" />

<col  class="org-left" />

<col  class="org-right" />
</colgroup>
<tbody>
<tr>
<td class="org-left">tt0111161</td>
<td class="org-right">9.3</td>
<td class="org-right">2651547</td>
<td class="org-left">The Shawshank Redemption</td>
<td class="org-right">1994</td>
</tr>

<tr>
<td class="org-left">tt0068646</td>
<td class="org-right">9.2</td>
<td class="org-right">1838044</td>
<td class="org-left">The Godfather</td>
<td class="org-right">1972</td>
</tr>

<tr>
<td class="org-left">tt0468569</td>
<td class="org-right">9.0</td>
<td class="org-right">2623735</td>
<td class="org-left">The Dark Knight</td>
<td class="org-right">2008</td>
</tr>

<tr>
<td class="org-left">tt0167260</td>
<td class="org-right">9.0</td>
<td class="org-right">1827464</td>
<td class="org-left">The Lord of the Rings: The Return of the King</td>
<td class="org-right">2003</td>
</tr>

<tr>
<td class="org-left">tt0108052</td>
<td class="org-right">9.0</td>
<td class="org-right">1343647</td>
<td class="org-left">Schindler's List</td>
<td class="org-right">1993</td>
</tr>

<tr>
<td class="org-left">tt0071562</td>
<td class="org-right">9.0</td>
<td class="org-right">1259465</td>
<td class="org-left">The Godfather Part II</td>
<td class="org-right">1974</td>
</tr>

<tr>
<td class="org-left">tt0050083</td>
<td class="org-right">9.0</td>
<td class="org-right">782903</td>
<td class="org-left">12 Angry Men</td>
<td class="org-right">1957</td>
</tr>

<tr>
<td class="org-left">tt0110912</td>
<td class="org-right">8.9</td>
<td class="org-right">2029684</td>
<td class="org-left">Pulp Fiction</td>
<td class="org-right">1994</td>
</tr>

<tr>
<td class="org-left">tt1375666</td>
<td class="org-right">8.8</td>
<td class="org-right">2325417</td>
<td class="org-left">Inception</td>
<td class="org-right">2010</td>
</tr>

<tr>
<td class="org-left">tt0137523</td>
<td class="org-right">8.8</td>
<td class="org-right">2096752</td>
<td class="org-left">Fight Club</td>
<td class="org-right">1999</td>
</tr>
</tbody>
</table>

That looks about right to me. There may be some small differences with the
current Top 10 list on IMDB if our data are a little stale.

It's early days still for Substrait, but it's exciting to see how far it's come
in the last 18 months!

## Why wouldn't I just use SQL for this?

It's a fair question. SQL _is_ everywhere, after all.

There are a few reasons we think you shouldn't ignore Substrait.

### Standards

SQL has a standard, but how closely do engines follow the standard? In our
experience, queries don't translate well between engines (this is one reason
Ibis exists!)

### Extensibility

Substrait is more extensible than SQL. Some DBMS have added in some very cool
features, but it usually involves diverging (sometimes widely) from the SQL
standard. Substrait has an extension system that allows plan producers and plan
consumers to agree on a well-typed and well-defined interaction that exists
outside of the core Substrait specification.

### Serialization and parsing

Parsing SQL can be a big pain (trust us). If you send a big string over the
wire, you need the engine on the other side to have a SQL parser to understand
what the message is. Now, obviously, SQL engines have those. But here, again,
standards (or lack of adherence to standards) can bite you. Extensibility is
also difficult here, because now the SQL parser needs to understand some new
custom syntax.

Protobuf is hardly a dream to work with, but it's a lot easier to consistently
define behavior AND to validate that behavior is correct. It's also smaller than
raw text.

## Wrapping Up

That's all for now! To quickly summarize:

Substrait is a new standard for representing relational algebra queries with
support in [Apache
Arrow](https://arrow.apache.org/docs/dev/cpp/streaming_execution.html#substrait),
[DuckDB](https://duckdb.org/docs/extensions/substrait),
[Velox](https://engineering.fb.com/2022/08/31/open-source/velox/), and more (and
more to come!).

Ibis can now generate substrait instead of string SQL, letting it take advantage
of this new standard.

Interested in substrait or ibis? Docs are available at

- [Substrait](https://substrait.io/)
- [Ibis Docs](https://ibis-project.org)

and the relevant GitHub repos are

- [Substrait GitHub](https://github.com/substrait-io/substrait)
- [Ibis Substrait GitHub](https://github.com/ibis-project/ibis-substrait)
- [Ibis GitHub](https://github.com/ibis-project/ibis)

Please feel free to reach out on GitHub!
