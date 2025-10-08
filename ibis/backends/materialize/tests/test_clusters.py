"""Tests for Materialize clusters.

Tests cover cluster creation, management, operations, and ALTER commands.
"""

from __future__ import annotations

import os

import ibis


class TestClusters:
    """Tests for cluster operations (create, drop, list).

    Clusters are computational resources for running queries and streaming objects.
    """

    def test_list_clusters(self, con):
        """Test listing clusters."""
        # Should return at least the default clusters
        clusters = con.list_clusters()
        assert isinstance(clusters, list)
        # Materialize has built-in clusters like 'quickstart'
        assert len(clusters) > 0
        assert "quickstart" in clusters

    def test_list_clusters_with_like(self, con):
        """Test listing clusters with LIKE pattern."""
        # Filter for quickstart cluster
        clusters = con.list_clusters(like="quick%")
        assert isinstance(clusters, list)
        assert "quickstart" in clusters

    def test_list_cluster_sizes(self, con):
        """Test listing available cluster sizes."""
        sizes = con.list_cluster_sizes()
        assert isinstance(sizes, list)
        assert len(sizes) > 0
        # Should contain at least some common sizes
        assert any("cc" in size or "C" in size for size in sizes)

    def test_current_cluster(self, con):
        """Test getting the current cluster."""
        current = con.current_cluster
        assert isinstance(current, str)
        assert len(current) > 0
        # Should be one of the available clusters
        clusters = con.list_clusters()
        assert current in clusters

    def test_set_cluster(self, con):
        """Test setting the active cluster."""
        # Get initial cluster
        initial = con.current_cluster

        # Set to quickstart (should always exist)
        con.set_cluster("quickstart")
        assert con.current_cluster == "quickstart"

        # Set back to initial
        con.set_cluster(initial)
        assert con.current_cluster == initial

    def test_connect_with_cluster_parameter(self):
        """Test connecting with cluster parameter."""

        # Create a new connection with cluster specified
        con = ibis.materialize.connect(
            host=os.environ.get("IBIS_TEST_MATERIALIZE_HOST", "localhost"),
            port=int(os.environ.get("IBIS_TEST_MATERIALIZE_PORT", "6875")),
            user=os.environ.get("IBIS_TEST_MATERIALIZE_USER", "materialize"),
            password=os.environ.get("IBIS_TEST_MATERIALIZE_PASSWORD", ""),
            database=os.environ.get("IBIS_TEST_MATERIALIZE_DATABASE", "materialize"),
            cluster="quickstart",
        )

        try:
            # Verify the cluster was set
            assert con.current_cluster == "quickstart"
        finally:
            con.disconnect()


class TestClusterAPI:
    """Tests documenting the cluster API.

    These tests demonstrate the API patterns without executing real cluster creation,
    since that would require actual compute resources and costs.
    """

    def test_basic_cluster_documented(self):
        """Document creating a basic cluster.

        >>> con.create_cluster("my_cluster", size="100cc")
        """

    def test_ha_cluster_documented(self):
        """Document creating a high-availability cluster with replicas.

        >>> con.create_cluster("ha_cluster", size="400cc", replication_factor=2)
        """

    def test_disk_cluster_documented(self):
        """Document creating a cluster with disk storage.

        >>> con.create_cluster("disk_cluster", size="200cc", disk=True)
        """

    def test_paused_cluster_documented(self):
        """Document creating an empty cluster (no replicas).

        Useful for creating clusters that will be started later:
        >>> con.create_cluster("paused_cluster", size="100cc", replication_factor=0)
        """

    def test_introspection_disabled_documented(self):
        """Document creating a cluster with introspection disabled.

        Disabling introspection can improve performance for simple workloads:
        >>> con.create_cluster("fast_cluster", size="100cc", introspection_interval="0")
        """

    def test_drop_cluster_documented(self):
        """Document dropping clusters.

        >>> con.drop_cluster("my_cluster", force=True)
        >>> con.drop_cluster("old_cluster", cascade=True)
        """

    def test_list_clusters_documented(self):
        """Document listing clusters.

        >>> con.list_clusters()
        ['quickstart', 'my_cluster', 'ha_cluster']

        >>> con.list_clusters(like="my%")
        ['my_cluster']
        """

    def test_list_cluster_sizes_documented(self, con):
        """Document listing available cluster sizes.

        >>> import ibis
        >>> con = ibis.materialize.connect()
        >>> sizes = con.list_cluster_sizes()
        >>> isinstance(sizes, list)
        True
        >>> len(sizes) > 0
        True
        >>> # Sizes typically follow patterns like '25cc', '50cc', etc.
        >>> any("cc" in size for size in sizes)
        True
        """


class TestAlterCommands:
    """Functional tests for ALTER commands."""

    def test_alter_cluster_rename(self, con):
        """Test renaming a cluster."""
        from ibis.util import gen_name

        cluster1 = gen_name("cluster")
        cluster2 = gen_name("cluster")

        try:
            # Create cluster
            con.create_cluster(cluster1, size="25cc", replication_factor=0)

            # Rename it
            con.alter_cluster(cluster1, rename_to=cluster2)

            # Verify new name exists
            clusters = con.list_clusters()
            assert cluster2 in clusters
            assert cluster1 not in clusters
        finally:
            con.drop_cluster(cluster1, force=True)
            con.drop_cluster(cluster2, force=True)

    def test_alter_cluster_set_options(self, con):
        """Test setting cluster options."""
        from ibis.util import gen_name

        cluster_name = gen_name("cluster")

        try:
            # Create cluster
            con.create_cluster(cluster_name, size="25cc", replication_factor=0)

            # Alter replication factor
            con.alter_cluster(cluster_name, set_options={"REPLICATION FACTOR": 1})

            # No error means success
            assert cluster_name in con.list_clusters()
        finally:
            con.drop_cluster(cluster_name, force=True)

    def test_alter_secret(self, con):
        """Test altering a secret's value."""
        from ibis.util import gen_name

        secret_name = gen_name("secret")

        try:
            # Create secret
            con.create_secret(secret_name, "initial_value")

            # Alter secret
            con.alter_secret(secret_name, "new_value")

            # Verify it still exists
            secrets = con.list_secrets()
            assert secret_name in secrets
        finally:
            con.drop_secret(secret_name, force=True)


class TestAlterCommandsAPI:
    """Documentation tests for ALTER commands."""

    def test_alter_cluster_rename_documented(self, con):
        """Document altering cluster name.

        >>> import ibis
        >>> from ibis.util import gen_name
        >>> con = ibis.materialize.connect()
        >>> cluster_old = gen_name("cluster_old")
        >>> cluster_new = gen_name("cluster_new")
        >>> try:
        ...     con.create_cluster(cluster_old, size="25cc", replication_factor=0)
        ...     con.alter_cluster(cluster_old, rename_to=cluster_new)
        ...     assert cluster_new in con.list_clusters()
        ... finally:
        ...     con.drop_cluster(cluster_old, force=True)
        ...     con.drop_cluster(cluster_new, force=True)
        """

    def test_alter_cluster_set_options_documented(self, con):
        """Document setting cluster options.

        >>> import ibis
        >>> from ibis.util import gen_name
        >>> con = ibis.materialize.connect()
        >>> cluster_name = gen_name("cluster")
        >>> try:
        ...     con.create_cluster(cluster_name, size="25cc", replication_factor=0)
        ...     con.alter_cluster(cluster_name, set_options={"REPLICATION FACTOR": 1})
        ...     assert cluster_name in con.list_clusters()
        ... finally:
        ...     con.drop_cluster(cluster_name, force=True)
        """

    def test_alter_cluster_reset_options_documented(self, con):
        """Document resetting cluster options.

        >>> import ibis
        >>> from ibis.util import gen_name
        >>> con = ibis.materialize.connect()
        >>> cluster_name = gen_name("cluster")
        >>> try:
        ...     con.create_cluster(cluster_name, size="25cc", replication_factor=2)
        ...     con.alter_cluster(cluster_name, reset_options=["REPLICATION FACTOR"])
        ...     assert cluster_name in con.list_clusters()
        ... finally:
        ...     con.drop_cluster(cluster_name, force=True)
        """

    def test_alter_secret_documented(self, con):
        """Document altering secrets.

        >>> import ibis
        >>> from ibis.util import gen_name
        >>> con = ibis.materialize.connect()
        >>> secret_name = gen_name("secret")
        >>> try:
        ...     con.create_secret(secret_name, "initial_value")
        ...     con.alter_secret(secret_name, "new_value")
        ...     assert secret_name in con.list_secrets()
        ... finally:
        ...     con.drop_secret(secret_name, force=True)
        """
