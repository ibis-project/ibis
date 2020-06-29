.. currentmodule:: ibis.bigquery.api

.. _bigquery:

Using Ibis with BigQuery
========================

To use the BigQuery client, you will need a Google Cloud Platform account.
Use the `BigQuery sandbox <https://cloud.google.com/bigquery/docs/sandbox>`__
to try the service for free.

The BigQuery client object
--------------------------

To use Ibis with BigQuery, you first must connect to BigQuery using the
:func:`ibis.bigquery.connect` function, optionally supplying Google API
credentials:

.. code-block:: python

   import ibis

   client = ibis.bigquery.connect(
       project_id=YOUR_PROJECT_ID,
       dataset_id='bigquery-public-data.stackoverflow'
   )

.. _bigquery-privacy:

Privacy
-------

This package is subject to the `NumFocus privacy policy
<https://numfocus.org/privacy-policy>`_. Your use of Google APIs with this
module is subject to each API's respective `terms of service
<https://developers.google.com/terms/>`_.

Google account and user data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Accessing user data
~~~~~~~~~~~~~~~~~~~

The :func:`~ibis.bigquery.api.connect` function provides access to data
stored in Google BigQuery and other sources such as Google Sheets or Cloud
Storage, via the federated query feature. Your machine communicates directly
with the Google APIs.

Storing user data
~~~~~~~~~~~~~~~~~

By default, your credentials are stored to a local file, such as
``~/.config/pydata/ibis.json``. All user data is stored on
your local machine. **Use caution when using this library on a shared
machine**.

Sharing user data
~~~~~~~~~~~~~~~~~

The BigQuery client only communicates with Google APIs. No user data is
shared with PyData, NumFocus, or any other servers.

Policies for application authors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Do not use the default client ID when using Ibis from an application,
library, or tool. Per the `Google User Data Policy
<https://developers.google.com/terms/api-services-user-data-policy>`_, your
application must accurately represent itself when authenticating to Google
API services.
