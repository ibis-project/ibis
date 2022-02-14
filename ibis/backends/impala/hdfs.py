# Copyright 2014 Cloudera Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file may adapt small portions of https://github.com/mtth/hdfs (MIT
# license), see the LICENSES directory.

"""File system module."""

from __future__ import annotations

import abc
import operator
import posixpath
import tarfile
from functools import wraps as implements
from typing import IO, Any, Literal

import ibis.common.exceptions as com
from ibis.config import options


class HDFSError(com.IbisError):
    """HDFS Error class."""


class HDFS:
    """Interface class to HDFS.

    Interface class to HDFS for ibis that abstracts away (and protects
    user/developer against) various 3rd party library API differences.
    """

    @abc.abstractmethod
    def exists(self, path: str) -> bool:
        """Check if the file exists.

        Parameters
        ----------
        path
            Path to a file on HDFS

        Returns
        -------
        bool
            Whether `path` exists.
        """
        raise NotImplementedError

    def status(self, path: str) -> dict:
        """Check if the status of the path.

        Parameters
        ----------
        path
            Path to a file on HDFS

        Returns
        -------
        dict
            Information about `path`
        """
        raise NotImplementedError

    def chmod(self, hdfs_path: str, permissions: str):
        """Change permissions of a file of directory.

        Parameters
        ----------
        hdfs_path
            Directory or path
        permissions
            Octal permissions string
        """
        raise NotImplementedError

    def chown(
        self,
        hdfs_path: str,
        owner: str | None = None,
        group: str | None = None,
    ) -> None:
        """Change owner (and/or group) of a file or directory.

        Parameters
        ----------
        hdfs_path
            Directory or path
        owner
            Name of owner
        group
            Name of group
        """
        raise NotImplementedError

    def head(
        self, hdfs_path: str, nbytes: int = 1024, offset: int = 0
    ) -> bytes:
        """Retrieve the requested number of bytes from a file.

        Parameters
        ----------
        hdfs_path
            Absolute HDFS path
        nbytes
            Number of bytes to retrieve
        offset
            Number of bytes at beginning of file to skip before retrieving data

        Returns
        -------
        bytes
            File data
        """
        raise NotImplementedError

    def get(
        self,
        hdfs_path: str,
        local_path: str,
        overwrite: bool = False,
    ) -> str:
        """Download remote file or directory to the local filesystem.

        Further keyword arguments are passed down to internal APIs.

        Parameters
        ----------
        hdfs_path
            Absolute HDFS path
        local_path
            Output location on the client
        overwrite
            Whether to overwrite the file on the client.

        Returns
        -------
        str
            The path to the written file or directory
        """
        raise NotImplementedError

    def put(
        self,
        hdfs_path: str,
        resource: str | bytes | IO,
        overwrite: bool = False,
        verbose: bool = None,
        **kwargs: Any,
    ) -> str:
        """Write a resource to HDFS.

        Further keyword arguments passed down to internal APIs.

        Parameters
        ----------
        hdfs_path
            Directory or path
        resource
            Relative or absolute path to local resource, or a file-like object
        overwrite
            Overwrite the HDFS file
        verbose
            Verbosity

        Returns
        -------
        str
            The path to the written file or directory
        """
        raise NotImplementedError

    def put_tarfile(
        self,
        hdfs_path: str,
        local_path: str,
        compression: Literal['gzip', 'bz2'] | None = 'gzip',
        verbose: bool = None,
        overwrite: bool = False,
    ) -> None:
        """Write the contents of a tar archive to HDFS.

        Parameters
        ----------
        hdfs_path
            Path on HDFS
        local_path
            Path to tar archive on the client
        compression
            Compression type. If `None` then no compression is done.
        overwrite
            Overwrite the HDFS file
        verbose
            Verbosity

        Raises
        ------
        ValueError
            if given compression is none of the following: None, gzip or bz2.
        """
        modes = {None: 'r', 'gzip': 'r:gz', 'bz2': 'r:bz2'}

        if compression not in modes:
            raise ValueError(f'Invalid compression type {compression}')
        mode = modes[compression]

        with tarfile.open(local_path, mode=mode) as tf:
            for info in filter(operator.methodcaller("isfile"), tf):
                buf = tf.extractfile(info)
                assert buf is not None
                abspath = posixpath.join(hdfs_path, info.path)
                self.put(abspath, buf, verbose=verbose, overwrite=overwrite)

    def put_zipfile(self, hdfs_path: str, local_path: str) -> None:
        """Write contents of zipfile archive to HDFS.

        Parameters
        ----------
        hdfs_path
            HDFS path
        local_path
            Client path to zip file
        """
        raise NotImplementedError

    def write(
        self,
        hdfs_path: str,
        buf: str | bytes | IO,
        overwrite: bool = False,
        blocksize: int | None = None,
        replication: int | None = None,
        buffersize: int | None = None,
    ) -> None:
        """Write bytes to HDFS.

        Parameters
        ----------
        hdfs_path : string
            HDFS path
        buf
            Data to write
        overwrite
            Overwrite data on HDFS
        blocksize
            HDFS block size
        replication
            Replication factor
        buffersize : int
            IO buffer size
        """
        raise NotImplementedError

    def mkdir(self, path: str) -> None:
        """Create a new directory.

        Parameters
        ----------
        path
            Directory to create
        """

    def ls(
        self,
        hdfs_path: str,
        status: bool = False,
    ) -> list[tuple[str, dict[str, Any]]]:
        """Return the contents of directory.

        Parameters
        ----------
        hdfs_path
            Directory
        status
            Give extended status information

        Returns
        -------
        list[tuple[str, dict[str, Any]]]
            Directory contents
        """
        raise NotImplementedError

    def size(self, hdfs_path: str) -> int:
        """Return the size of a file or directory.

        Parameters
        ----------
        hdfs_path
            Path to HDFS file or directory

        Returns
        -------
        int
            Size of a file or directory
        """
        raise NotImplementedError

    def tail(self, hdfs_path: str, nbytes: int = 1024) -> bytes:
        """Retrieve `nbytes` bytes from the end of a file.

        Parameters
        ----------
        hdfs_path
            Path on HDFS
        nbytes
            Number of bytes to read

        Returns
        -------
        bytes
            Bytes from the end of `hdfs_path`
        """
        raise NotImplementedError

    def mv(
        self,
        hdfs_path_src: str,
        hdfs_path_dest: str,
        overwrite: bool = True,
    ) -> None:
        """Move `hdfs_path_src` to `hdfs_path_dest`.

        Parameters
        ----------
        hdfs_path_src
            Source path
        hdfs_path_dest
            Destination path
        overwrite
            Overwrite the destination path if it exists.
        """
        raise NotImplementedError

    def cp(self, hdfs_path_src: str, hdfs_path_dest: str) -> None:
        """Copy `hdfs_path_src` to `hdfs_path_dest`.

        Parameters
        ----------
        hdfs_path_src
            Source path
        hdfs_path_dest
            Destination path
        """
        raise NotImplementedError

    def delete(self, path: str) -> None:
        """Delete a single file.

        Parameters
        ----------
        path
            Path to delete
        """

    def rm(self, path: str) -> None:
        """Remove a single file.

        Parameters
        ----------
        path
            Path to remove
        """
        self.delete(path)

    client: Any

    def rmdir(self, path: str) -> None:
        """Delete a directory and all of its contents.

        Parameters
        ----------
        path
            Directory to remove
        """
        self.client.delete(path, recursive=True)

    def _find_any_file(self, hdfs_dir: str) -> str:
        contents = self.ls(hdfs_dir, status=True)

        def valid_filename(name):
            head, tail = posixpath.split(name)

            tail = tail.lower()
            return (
                not tail.endswith('.tmp')
                and not tail.endswith('.copying')
                and not tail.startswith('_')
                and not tail.startswith('.')
            )

        for filename, meta in contents:
            if meta['type'].lower() == 'file' and valid_filename(filename):
                return filename
        raise com.IbisError('No files found in the passed directory')


class WebHDFS(HDFS):
    """A WebHDFS-based interface to HDFS using the HDFSCli library."""

    def __init__(self, client) -> None:
        self.client = client

    @property
    def protocol(self) -> Literal['webhdfs']:
        """Return the protocol used by WebHDFS."""
        return 'webhdfs'

    def status(self, path: str) -> dict:
        return self.client.status(path)

    @implements(HDFS.chmod)
    def chmod(self, path: str, permissions: str):
        self.client.set_permission(path, permissions)

    @implements(HDFS.chown)
    def chown(self, path: str, owner=None, group=None):
        self.client.set_owner(path, owner, group)

    @implements(HDFS.exists)
    def exists(self, path: str) -> bool:
        return not self.client.status(path, strict=False) is None

    @implements(HDFS.ls)
    def ls(self, hdfs_path: str, status: bool = False) -> list:
        return self.client.list(hdfs_path, status=status)

    @implements(HDFS.mkdir)
    def mkdir(self, dir_path: str):
        self.client.makedirs(dir_path)

    @implements(HDFS.size)
    def size(self, hdfs_path: str) -> int:
        return self.client.content(hdfs_path)['length']

    @implements(HDFS.mv)
    def mv(
        self,
        hdfs_path_src: str,
        hdfs_path_dest: str,
        overwrite: bool = True,
    ) -> None:
        if overwrite and self.exists(hdfs_path_dest):
            if self.status(hdfs_path_dest)['type'] == 'FILE':
                self.rm(hdfs_path_dest)
        self.client.rename(hdfs_path_src, hdfs_path_dest)

    def delete(self, hdfs_path: str, recursive: bool = False) -> bool:
        return self.client.delete(hdfs_path, recursive=recursive)

    @implements(HDFS.head)
    def head(
        self,
        hdfs_path: str,
        nbytes: int = 1024,
        offset: int = 0,
    ) -> bytes:
        with self.client.read(
            hdfs_path,
            offset=offset,
            length=nbytes,
        ) as reader:
            return reader.read()

    @implements(HDFS.put)
    def put(
        self,
        hdfs_path: str,
        resource: str | bytes | IO,
        overwrite: bool = False,
        verbose: bool = None,
        **kwargs: Any,
    ):
        verbose = verbose or options.verbose
        if isinstance(resource, str):
            # `resource` is a path.
            return self.client.upload(
                hdfs_path, resource, overwrite=overwrite, **kwargs
            )
        else:
            # `resource` is a file-like object.
            hdfs_path = self.client.resolve(hdfs_path)
            self.client.write(
                hdfs_path, data=resource, overwrite=overwrite, **kwargs
            )
            return hdfs_path

    @implements(HDFS.get)
    def get(
        self,
        hdfs_path: str,
        local_path: str,
        overwrite: bool = False,
        **kwargs: Any,
    ) -> str:
        return self.client.download(
            hdfs_path, local_path, overwrite=overwrite, **kwargs
        )


def hdfs_connect(
    host: str = 'localhost',
    port: int = 50070,
    protocol: Literal['webhdfs'] = 'webhdfs',
    use_https: str = 'default',
    auth_mechanism: str = 'NOSASL',
    verify: bool = True,
    session: Any = None,
    **kwds: Any,
) -> WebHDFS:
    """Connect to HDFS.

    Parameters
    ----------
    host
        Host name of the HDFS NameNode
    port
        NameNode's WebHDFS port
    protocol
        The protocol used to communicate with HDFS. The only valid value is
        ``'webhdfs'``.
    use_https
        Connect to WebHDFS with HTTPS, otherwise plain HTTP. For secure
        authentication, the default for this is True, otherwise False.
    auth_mechanism
        Set to NOSASL or PLAIN for non-secure clusters.
        Set to GSSAPI or LDAP for Kerberos-secured clusters.
    verify
        Set to `False` to turn off verifying SSL certificates.
    session
        A custom `requests.Session` object.

    Returns
    -------
    WebHDFS
        WebHDFS client
    """
    import requests

    if session is None:
        session = requests.Session()
    session.verify = verify
    if auth_mechanism in ('GSSAPI', 'LDAP'):
        from hdfs.ext.kerberos import KerberosClient

        if use_https == 'default':
            prefix = 'https'
        else:
            prefix = 'https' if use_https else 'http'

        # note SSL
        url = f'{prefix}://{host}:{port}'
        kwds.setdefault('mutual_auth', 'OPTIONAL')
        hdfs_client = KerberosClient(url, session=session, **kwds)
    else:
        if use_https == 'default':
            prefix = 'http'
        else:
            prefix = 'https' if use_https else 'http'
        from hdfs.client import InsecureClient

        url = f'{prefix}://{host}:{port}'
        hdfs_client = InsecureClient(url, session=session, **kwds)
    return WebHDFS(hdfs_client)
