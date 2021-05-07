"""File system module."""
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

import posixpath
from functools import wraps as implements

import ibis.common.exceptions as com
from ibis.config import options


class HDFSError(com.IbisError):
    """HDFS Error class."""

    pass


class HDFS:
    """Interface class to HDFS.

    Interface class to HDFS for ibis that abstracts away (and protects
    user/developer against) various 3rd party library API differences.
    """

    def log(self, message: str):
        """Print a log message.

        Parameters
        ----------
        message: string
        """
        print(message)

    def exists(self, path: str) -> bool:
        """Check if the file exists.

        Parameters
        ----------
        path : string

        Returns
        -------
        bool

        Raises
        ------
        NotImplementedError

        """
        raise NotImplementedError

    def status(self, path: str) -> dict:
        """Check if the status of the path.

        Parameters
        ----------
        path : string

        Returns
        -------
        status : dict

        Raises
        ------
        NotImplementedError

        """
        raise NotImplementedError

    def chmod(self, hdfs_path: str, permissions: str):
        """Change permissions of a file of directory.

        Parameters
        ----------
        hdfs_path : string
          Directory or path
        permissions : string
          Octal permissions string

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError

    def chown(self, hdfs_path: str, owner: str = None, group: str = None):
        """Change owner (and/or group) of a file or directory.

        Parameters
        ----------
        hdfs_path : string
          Directory or path
        owner : string, optional
          Name of owner
        group : string, optional
          Name of group

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError

    def head(
        self, hdfs_path: str, nbytes: int = 1024, offset: int = 0
    ) -> bytes:
        """Retrieve the requested number of bytes from a file.

        Parameters
        ----------
        hdfs_path : string
          Absolute HDFS path
        nbytes : int, default 1024 (1K)
          Number of bytes to retrieve
        offset : int, default 0
          Number of bytes at beginning of file to skip before retrieving data

        Returns
        -------
        head_data : bytes

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError

    def get(
        self, hdfs_path: str, local_path: str = '.', overwrite: bool = False
    ) -> str:
        """
        Download remote file or directory to the local filesystem.

        Parameters
        ----------
        hdfs_path : string
        local_path : string, default '.'
        overwrite : bool, default False

        Further keyword arguments passed down to any internal API used.

        Returns
        -------
        written_path : string
          The path to the written file or directory

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError

    def put(
        self,
        hdfs_path: str,
        resource,
        overwrite: bool = False,
        verbose: bool = None,
        **kwargs,
    ) -> str:
        """
        Write file or directory to HDFS.

        Parameters
        ----------
        hdfs_path : string
          Directory or path
        resource : string or buffer-like
          Relative or absolute path to local resource, or a file-like object
        overwrite : boolean, default False
        verbose : boolean, default ibis options.verbose

        Further keyword arguments passed down to any internal API used.

        Returns
        -------
        written_path : string
          The path to the written file or directory

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError

    def put_tarfile(
        self,
        hdfs_path: str,
        local_path: str,
        compression: str = 'gzip',
        verbose: bool = None,
        overwrite: bool = False,
    ):
        """
        Write contents of tar archive to HDFS.

        Write contents of tar archive to HDFS directly without having to
        decompress it locally first.

        Parameters
        ----------
        hdfs_path : string
        local_path : string
        compression : {'gzip', 'bz2', None}
        overwrite : boolean, default False
        verbose : boolean, default None (global default)

        Raises
        ------
        ValueError
            if given compression is none of the following: None, gzip or bz2.
        """
        import tarfile

        modes = {None: 'r', 'gzip': 'r:gz', 'bz2': 'r:bz2'}

        if compression not in modes:
            raise ValueError(
                'Invalid compression type {0}'.format(compression)
            )
        mode = modes[compression]

        tf = tarfile.open(local_path, mode=mode)
        for info in tf:
            if not info.isfile():
                continue

            buf = tf.extractfile(info)
            abspath = posixpath.join(hdfs_path, info.path)
            self.put(abspath, buf, verbose=verbose, overwrite=overwrite)

    def put_zipfile(self, hdfs_path: str, local_path: str):
        """Write contents of zipfile archive to HDFS.

        Parameters
        ----------
        hdfs_path : string
        local_path : string

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError

    def write(
        self,
        hdfs_path: str,
        buf,
        overwrite: bool = False,
        blocksize: int = None,
        replication=None,
        buffersize: int = None,
    ):
        """HDFS Write function.

        Parameters
        ----------
        hdfs_path : string
        buf
        overwrite : bool, defaul False
        blocksize : int
        replication
        buffersize : int

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError

    def mkdir(self, path: str):
        """Create new directory.

        Parameters
        ----------
        path : string
        """
        pass

    def ls(self, hdfs_path: str, status: bool = False) -> list:
        """Return contents of directory.

        Parameters
        ----------
        hdfs_path : string
        status : bool

        Returns
        -------
        list

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError

    def size(self, hdfs_path: str) -> int:
        """Return total size of file or directory.

        Parameters
        ----------
        hdfs_path : basestring

        Returns
        -------
        size : int

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError

    def tail(self, hdfs_path: str, nbytes: int = 1024) -> bytes:
        """Retrieve the requested number of bytes from the end of a file.

        Parameters
        ----------
        hdfs_path : string
        nbytes : int

        Returns
        -------
        data_tail : bytes

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError

    def mv(
        self, hdfs_path_src: str, hdfs_path_dest: str, overwrite: bool = True
    ):
        """Move hdfs_path_src to hdfs_path_dest.

        Parameters
        ----------
        hdfs_path_src: string
        hdfs_path_dest: string
        overwrite : boolean, default True
          Overwrite hdfs_path_dest if it exists.

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError

    def cp(self, hdfs_path_src: str, hdfs_path_dest: str):
        """Copy hdfs_path_src to hdfs_path_dest.

        Parameters
        ----------
        hdfs_path_src : string
        hdfs_path_dest : string

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError

    def rm(self, path: str):
        """Delete a single file.

        Parameters
        ----------
        path : string
        """
        return self.delete(path)

    def rmdir(self, path: str):
        """Delete a directory and all its contents.

        Parameters
        ----------
        path : string
        """
        self.client.delete(path, recursive=True)

    def _find_any_file(self, hdfs_dir):
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

    def __init__(self, client):
        self.client = client

    @property
    def protocol(self) -> str:
        """Return the protocol used by WebHDFS.

        Returns
        -------
        protocol : string
        """
        return 'webhdfs'

    def status(self, path: str) -> dict:
        """Retrieve HDFS metadata for path.

        Parameters
        ----------
        path : str

        Returns
        -------
        status : dict
            Client status
        """
        return self.client.status(path)

    @implements(HDFS.chmod)
    def chmod(self, path: str, permissions: str):
        """Change the permissions of a HDFS file.

        Parameters
        ----------
        path : string
        permissions : string
            New octal permissions string of the file.
        """
        self.client.set_permission(path, permissions)

    @implements(HDFS.chown)
    def chown(self, path: str, owner=None, group=None):
        """
        Change the owner of a HDFS file.

        At least one of `owner` and `group` must be specified.

        Parameters
        ----------
        hdfs_path : HDFS path.
        owner : string, optional
        group: string, optional
        """
        self.client.set_owner(path, owner, group)

    @implements(HDFS.exists)
    def exists(self, path: str) -> dict:
        """Check if the HDFS file exists.

        Parameters
        ----------
        path : string

        Returns
        -------
        bool
        """
        return not self.client.status(path, strict=False) is None

    @implements(HDFS.ls)
    def ls(self, hdfs_path: str, status: bool = False) -> list:
        """Return contents of directory.

        Parameters
        ----------
        hdfs_path : string
        status : bool

        Returns
        -------
        list
        """
        return self.client.list(hdfs_path, status=status)

    @implements(HDFS.mkdir)
    def mkdir(self, dir_path: str):
        """Create new directory.

        Parameters
        ----------
        path : string
        """
        self.client.makedirs(dir_path)

    @implements(HDFS.size)
    def size(self, hdfs_path: str) -> int:
        """Return total size of file or directory.

        Parameters
        ----------
        hdfs_path : string

        Returns
        -------
        size : int
        """
        return self.client.content(hdfs_path)['length']

    @implements(HDFS.mv)
    def mv(
        self, hdfs_path_src: str, hdfs_path_dest: str, overwrite: bool = True
    ):
        """Move hdfs_path_src to hdfs_path_dest.

        Parameters
        ----------
        hdfs_path_src: string
        hdfs_path_dest: string
        overwrite : boolean, default True
          Overwrite hdfs_path_dest if it exists.
        """
        if overwrite and self.exists(hdfs_path_dest):
            if self.status(hdfs_path_dest)['type'] == 'FILE':
                self.rm(hdfs_path_dest)
        self.client.rename(hdfs_path_src, hdfs_path_dest)

    def delete(self, hdfs_path: str, recursive: bool = False) -> bool:
        """Delete a file located at `hdfs_path`.

        Parameters
        ----------
        hdfs_path : string
        recursive : bool, default False

        Returns
        -------
        bool
            True if the function was successful.
        """
        return self.client.delete(hdfs_path, recursive=recursive)

    @implements(HDFS.head)
    def head(
        self, hdfs_path: str, nbytes: int = 1024, offset: int = 0
    ) -> bytes:
        """Retrieve the requested number of bytes from a file.

        Parameters
        ----------
        hdfs_path : string
          Absolute HDFS path
        nbytes : int, default 1024 (1K)
          Number of bytes to retrieve
        offset : int, default 0
          Number of bytes at beginning of file to skip before retrieving data

        Returns
        -------
        head_data : bytes
        """
        _reader = self.client.read(hdfs_path, offset=offset, length=nbytes)
        with _reader as reader:
            return reader.read()

    @implements(HDFS.put)
    def put(
        self,
        hdfs_path: str,
        resource,
        overwrite: bool = False,
        verbose: bool = None,
        **kwargs,
    ):
        """
        Write file or directory to HDFS.

        Parameters
        ----------
        hdfs_path : string
          Directory or path
        resource : string or buffer-like
          Relative or absolute path to local resource, or a file-like object
        overwrite : boolean, default False
        verbose : boolean, default ibis options.verbose

        Further keyword arguments passed down to any internal API used.

        Returns
        -------
        written_path : string
          The path to the written file or directory
        """
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
        verbose: bool = None,
        **kwargs,
    ) -> str:
        """
        Download remote file or directory to the local filesystem.

        Parameters
        ----------
        hdfs_path : string
        local_path : string, default '.'
        overwrite : bool, default False

        Further keyword arguments passed down to any internal API used.

        Returns
        -------
        written_path : string
          The path to the written file or directory
        """
        verbose = verbose or options.verbose
        return self.client.download(
            hdfs_path, local_path, overwrite=overwrite, **kwargs
        )


def hdfs_connect(
    host='localhost',
    port=50070,
    protocol='webhdfs',
    use_https='default',
    auth_mechanism='NOSASL',
    verify=True,
    session=None,
    **kwds,
):
    """Connect to HDFS.

    Parameters
    ----------
    host : str
        Host name of the HDFS NameNode
    port : int
        NameNode's WebHDFS port
    protocol : str,
        The protocol used to communicate with HDFS. The only valid value is
        ``'webhdfs'``.
    use_https : bool
        Connect to WebHDFS with HTTPS, otherwise plain HTTP. For secure
        authentication, the default for this is True, otherwise False.
    auth_mechanism : str
        Set to NOSASL or PLAIN for non-secure clusters.
        Set to GSSAPI or LDAP for Kerberos-secured clusters.
    verify : bool
        Set to :data:`False` to turn off verifying SSL certificates.
    session : Optional[requests.Session]
        A custom :class:`requests.Session` object.

    Notes
    -----
    Other keywords are forwarded to HDFS library classes.

    Returns
    -------
    WebHDFS

    """
    import requests

    if session is None:
        session = requests.Session()
    session.verify = verify
    if auth_mechanism in ('GSSAPI', 'LDAP'):
        if use_https == 'default':
            prefix = 'https'
        else:
            prefix = 'https' if use_https else 'http'
        try:
            import requests_kerberos  # noqa: F401
        except ImportError:
            raise com.IbisError(
                "Unable to import requests-kerberos, which is required for "
                "Kerberos HDFS support. Install it by executing `pip install "
                "requests-kerberos` or `pip install hdfs[kerberos]`."
            )
        from hdfs.ext.kerberos import KerberosClient

        # note SSL
        url = '{0}://{1}:{2}'.format(prefix, host, port)
        kwds.setdefault('mutual_auth', 'OPTIONAL')
        hdfs_client = KerberosClient(url, session=session, **kwds)
    else:
        if use_https == 'default':
            prefix = 'http'
        else:
            prefix = 'https' if use_https else 'http'
        from hdfs.client import InsecureClient

        url = '{}://{}:{}'.format(prefix, host, port)
        hdfs_client = InsecureClient(url, session=session, **kwds)
    return WebHDFS(hdfs_client)
