.. Ibis documentation master file, created by
   sphinx-quickstart on Wed Jun 10 11:06:29 2015.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Ibis Documentation
==================

Ibis is a new Python data analysis framework with the goal of enabling data
scientists and data engineers to be as productive working with big data as they
are working with small and medium data today. In doing so, we will enable
Python to become a true first-class language for Apache Hadoop, without
compromises in functionality, usability, or performance. Having spent much of
the last decade improving the usability of the single-node Python experience
(with pandas and other projects), we are looking to achieve:

* 100% Python end-to-end user workflows
* Native hardware speeds for a broad set of use cases
* Full-fidelity data analysis without extractions or sampling
* Scalability for big data
* Integration with the existing Python data ecosystem (pandas, scikit-learn,
  NumPy, and so on)

Ibis is being designed to take advantage of architectural synergies with the
`Impala project <http://impala.io/>`_ that will enable high performance Python
at massive scale without serialization or other interface
bottlenecks. Specifically, we have on the roadmap:

* Support for Impala's forthcoming **complex types**: lists, maps, and structs
  as first-class value types.
* Fast Python API for a canonical **in-memory columnar data format** being
  developed for Impala and to be standardized amongst software components.
* Enabling intepreted Python user-defined functions to be run on Impala nodes
  and perform computations directly on columnar data in shared memory without
  any need for deserialization. This will enable users to leverage the
  **existing Python data ecosystem**, both tools and libraries, at performance
  and scale never seen before.
* Expanding the useful set of Python that can be translated to LLVM IR to
  achieve true **native performance at scale** on complex data within Impala.
* Exposing **machine learning functionality** already available in MADLib.

This current version of Ibis includes a great deal of useful big data
functionality, putting Impala, the open source interactive SQL-on-Hadoop
engine, right at your fingertips in Python:

* A pandas-like data expression system providing comprehensive coverage of the
  functionality already provided by Impala. It is composable and semantically
  complete; if you can write it with SQL, you can write it with Ibis, often
  with substantially less code. This includes such tricky constructs as

  * Window functions
  * Correlated and uncorrelated subqueries
  * Self-joins

* High level analytics tools like bucketing, top-k, histogram, and
  value_counts.
* Tools for performing computations directly on datasets in HDFS, hiding the
  low-level details of Impala for accessing such data.
* Tools to simplify interactions with HDFS
* Interoperability with pandas: executing expressions returns pandas objects,
  and pandas objects can be written back to HDFS (experimental).

Please stay tuned to http://ibis-project.org

Since this is a young project, the documentation is definitely patchy in
places, but this will improve as things progress.

.. toctree::
   :maxdepth: 1

   getting-started
   configuration
   api
   release
   developer
   type-system
   legal

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
