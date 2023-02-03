from __future__ import annotations

import pyarrow as pa


class IbisRecordBatchReader(pa.ipc.RecordBatchReader):
    """Hack to make sure the database cursor isn't garbage collected.

    Without this hack batches are streamed out of the RecordBatchReader on a
    closed cursor.
    """

    def __init__(self, reader, cursor):
        self.reader = reader
        self.cursor = cursor

    def close(self):
        self.reader.close()
        del self.cursor

    def read_all(self):
        return self.reader.read_all()

    def read_next_batch(self):
        return self.reader.read_next_batch()

    def read_pandas(self):
        return self.reader.read_pandas()

    @property
    def schema(self):
        return self.reader.schema
