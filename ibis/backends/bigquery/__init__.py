import google.auth.credentials
import google.cloud.bigquery  # noqa: F401, fail early if bigquery is missing
import pydata_google_auth

import ibis.config
import ibis.backends.base

from .client import BigQueryClient
from .compiler import BigQueryQueryBuilder, BigQueryDialect

try:
    from .udf import udf
except ImportError:
    pass


SCOPES = ["https://www.googleapis.com/auth/bigquery"]
CLIENT_ID = (
    "546535678771-gvffde27nd83kfl6qbrnletqvkdmsese.apps.googleusercontent.com"
)
CLIENT_SECRET = "iU5ohAF2qcqrujegE3hQ1cPt"


class Backend(ibis.backends.base.BaseBackend):
    name = 'bigquery'
    builder = BigQueryQueryBuilder
    dialect = BigQueryDialect

    def connect(self):
        """Create a BigQueryClient for use with Ibis.

        Parameters
        ----------
        project_id : str
            A BigQuery project id.
        dataset_id : str
            A dataset id that lives inside of the project indicated by
            `project_id`.
        credentials : google.auth.credentials.Credentials
        application_name : str
            A string identifying your application to Google API endpoints.

        Returns
        -------
        BigQueryClient

        """
        if credentials is None:
            credentials_cache = pydata_google_auth.cache.ReadWriteCredentialsCache(
                filename="ibis.json"
            )
            credentials, project_id = pydata_google_auth.default(
                SCOPES,
                client_id=CLIENT_ID,
                client_secret=CLIENT_SECRET,
                credentials_cache=credentials_cache,
            )

        return BigQueryClient(
            project_id,
            dataset_id=dataset_id,
            credentials=credentials,
            application_name=application_name,
        )

    def register_options(self):
        ibis.config.register_option('partition_col', 'PARTITIONTIME')
