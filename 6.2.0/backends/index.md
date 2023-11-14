# Backends

A backend is where execution of Ibis table expressions occur after compiling into some intermediate representation. A backend is often a database and the intermediate representation often SQL, but several types of backends exist.

See the [configuration guide](../how_to/configuration.md#default-backend)
to inspect or reconfigure the backend used by default. View the [operation support matrix](_support_matrix.md) to see which operations
are supported by each backend.

Each backend has its own configuration options documented here.
