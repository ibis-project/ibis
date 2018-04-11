MapD IBIS backend
=================

In this document it would be explained the main aspects of `ibis` backend and
`MapD ibis` backend implementation.

Modules
-------

MapD backend has 5 modules, which 3 of them are the main modules:

- `api`
- `client`
- `compiler`

The `identifiers` and `operations` modules were created to organize `compiler`
method.

api
---

`api` module is the central key to access the backend. Ensure to include
the follow code into `ibi.__init__`:

.. code-block:: python

    with suppress(ImportError):
        # pip install ibis-framework[mapd]
        import ibis.mapd.api as mapd

Basically, there is 3 function on `api` module>

- `compile`
- `connect`
- `verify`

`compile` method compiles a `ibis` expression to `SQL`:

.. code-block:: python

    t = mapd_cli.table('flights_2008_10k')
    proj = t['arrtime', 'arrdelay']
    print(ibis.mapd.compile(proj))

`connect` method instantiates a `MapDClient` object that connect to the `MapD`
database specified:

.. code-block:: python

    mapd_cli = ibis.mapd.connect(
        host='localhost', user='mapd', password='HyperInteractive',
        port=9091, dbname='mapd'
    )

`verify` method checks if the `ibis` expression can be compiled.

.. code-block:: python

    t = mapd_cli.table('flights_2008_10k')
    proj = t['arrtime', 'arrdelay']
    assert ibis.mapd.verify(proj) == True

client
------

`client` module has the main classes to handle connection to the `MapD`
database.

The main classes are:

- `MapDClient`
- `MapDQuery`
- `MapDDataType`
- `MapDCursor`

`MapDDataType` class is used to translate data type from `ibis` and to `ibis`.
Its main methods are:

- `parse`
- `to_ibis`
- `from_ibis`

`parse` method ... # @TODO

`to_ibis` method ... # @TODO

`from_ibis` method ... # @TODO

`MapDClient` class is used to connect to `MapD` database and manipulation data
expression. Its main methods are:

- __init__
- _build_ast
- _execute
- _fully_qualified_name
- _get_table_schema
- _table_expr_klass
- log
- close
- database
- current_database
- set_database
- exists_database
- list_databases
- exists_table
- list_tables
- get_schema
- version

`_build_ast` method is required.

`MapDQuery` class should be used redefine at least `_fetch` method. If `Query`
class is used instead, when `MapDClient.execute` method is called, a exception
is raised.

    (...) once the data arrives from the database we need to convert that data
    to a pandas DataFrame.

    The Query class, with its _fetch() method, provides a way for ibis
    SQLClient objects to do any additional processing necessary after
    the database returns results to the client.
    (http://docs.ibis-project.org/design.html#execution)

`MapDCursor` class was created just to allow `ibis.client.Query.execute`
useful automatically, because it uses `with` statement:

.. code-block:: Python
    with self.client._execute(self.compiled_ddl, results=True) as cur:
       ...

Otherwise, `MapDQuery` should rewrites `execute` method with no `with`
statement.

compiler
--------

The main classes inside `compiler` module are:

- MapDDialect
- MapDExprTranslator
- MapDQueryBuilder
- MapDSelect
- MapDSelectBuilder
- MapDTableSetFormatter

operations
----------

    `Node` subclasses make up the core set of operations of ibis.
    Each node corresponds to a particular operation.
    Most nodes are defined in the `operations` module.
    (http://docs.ibis-project.org/design.html#the-node-class).


identifiers
-----------

`identifiers` module keep a set of identifiers (`_identifiers`) to be used
inside `quote_identifier` function (inside the same module). `_identifiers` is
a set of reserved words from `MapD` language.

`quote_identifiers` is used to put quotes around the string sent if the string
match to specific criteria.

References
----------

- ibis API: http://docs.ibis-project.org/api.html
