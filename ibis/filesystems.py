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

from os import path as osp
import os
import posixpath
import shutil

import six

from ibis.config import options
import ibis.common as com
import ibis.util as util

from hdfs.util import temppath


class HDFSError(com.IbisError):
    pass


def implements(f):
    def decorator(g):
        g.__doc__ = f.__doc__
        return g
    return decorator


class HDFS(object):

    """
    Interface class to HDFS for ibis that abstracts away (and protects
    user/developer against) various 3rd party library API differences.
    """
    def log(self, message):
        print(message)

    def exists(self, path):
        raise NotImplementedError

    def status(self, path):
        raise NotImplementedError

    def head(self, hdfs_path, nbytes=1024, offset=0):
        raise NotImplementedError

    def get(self, hdfs_path, local_path='.', overwrite=False):
        """
        Download remote file or directory to the local filesystem

        Parameters
        ----------
        hdfs_path : string
        local_path : string, default '.'
        """
        raise NotImplementedError

    def put(self, hdfs_path, resource, overwrite=False, verbose=None,
            **kwargs):
        """
        Write file or directory to HDFS

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
        raise NotImplementedError

    def write(self, hdfs_path, buf, overwrite=False, blocksize=None,
              replication=None, buffersize=None):
        raise NotImplementedError

    def mkdir(self, path, create_parent=False):
        pass

    def ls(self, hdfs_path, status=False):
        """
        Return contents of directory

        Parameters
        ----------
        hdfs_path : string
        """
        raise NotImplementedError

    def tail(self, hdfs_path, nbytes=1024):
        raise NotImplementedError

    def rm(self, path):
        return self.delete(path)

    def rmdir(self, path):
        self.client.delete(path, recursive=True)

    def find_any_file(self, hdfs_dir):
        contents = self.ls(hdfs_dir, status=True)

        def valid_filename(name):
            head, tail = posixpath.split(name)

            tail = tail.lower()
            return (not tail.endswith('.tmp') and
                    not tail.endswith('.copying') and
                    not tail.startswith('_') and
                    not tail.startswith('.'))

        for filename, meta in contents:
            if meta['type'].lower() == 'file' and valid_filename(filename):
                return filename
        raise com.IbisError('No files found in the passed directory')


class WebHDFS(HDFS):

    """
    A WebHDFS-based interface to HDFS using the HDFSCli library
    """

    def __init__(self, client):
        self.client = client

    @property
    def protocol(self):
        return 'webhdfs'

    def status(self, path):
        return self.client.status(path)

    @implements(HDFS.exists)
    def exists(self, path):
        try:
            self.client.status(path)
            return True
        except Exception:
            return False

    @implements(HDFS.ls)
    def ls(self, hdfs_path, status=False):
        contents = self.client.list(hdfs_path)
        if not status:
            return [path for path, detail in contents]
        else:
            return contents

    @implements(HDFS.mkdir)
    def mkdir(self, dir_path, create_parent=False):
        # ugh, see #252

        # create a temporary file, then delete it
        dummy = posixpath.join(dir_path, util.guid())
        self.client.write(dummy, '')
        self.client.delete(dummy)

    def delete(self, hdfs_path, recursive=False):
        """

        """
        return self.client.delete(hdfs_path, recursive=recursive)

    def head(self, hdfs_path, nbytes=1024, offset=0):
        gen = self.client.read(hdfs_path, offset=offset, length=nbytes)
        return ''.join(gen)

    @implements(HDFS.put)
    def put(self, hdfs_path, resource, overwrite=False, verbose=None,
            **kwargs):
        verbose = verbose or options.verbose
        is_path = isinstance(resource, six.string_types)

        if is_path and osp.isdir(resource):
            for dirpath, dirnames, filenames in os.walk(resource):
                rel_dir = osp.relpath(dirpath, resource)
                if rel_dir == '.':
                    rel_dir = ''
                for fpath in filenames:
                    abs_path = osp.join(dirpath, fpath)
                    rel_hdfs_path = posixpath.join(hdfs_path, rel_dir, fpath)
                    self.put(rel_hdfs_path, abs_path, overwrite=overwrite,
                             verbose=verbose, **kwargs)
        else:
            if is_path:
                if verbose:
                    self.log('Writing local {0} to HDFS {1}'.format(resource,
                                                                    hdfs_path))
                self.client.upload(hdfs_path, resource,
                                   overwrite=overwrite, **kwargs)
            else:
                if verbose:
                    self.log('Writing buffer to HDFS {0}'.format(hdfs_path))
                self.client.write(hdfs_path, resource, overwrite=overwrite,
                                  **kwargs)

    @implements(HDFS.get)
    def get(self, hdfs_path, local_path, overwrite=False, verbose=None):
        verbose = verbose or options.verbose

        hdfs_path = hdfs_path.rstrip(posixpath.sep)

        if osp.isdir(local_path) and not overwrite:
            dest = osp.join(local_path, posixpath.basename(hdfs_path))
        else:
            local_dir = osp.dirname(local_path) or '.'
            if osp.isdir(local_dir):
                dest = local_path
            else:
                # fail early
                raise HDFSError('Parent directory %s does not exist',
                                local_dir)

        # TODO: threadpool

        def _get_file(remote, local):
            if verbose:
                self.log('Writing HDFS {0} to local {1}'.format(remote, local))
            self.client.download(remote, local, overwrite=overwrite)

        def _scrape_dir(path, dst):
            objs = self.client.list(path)
            for hpath, detail in objs:
                relpath = posixpath.relpath(hpath, hdfs_path)
                full_opath = posixpath.join(dst, relpath)

                if detail['type'] == 'FILE':
                    _get_file(hpath, full_opath)
                else:
                    os.makedirs(full_opath)
                    _scrape_dir(hpath, dst)

        status = self.status(hdfs_path)
        if status['type'] == 'FILE':
            if not overwrite and osp.exists(local_path):
                raise IOError('{0} exists'.format(local_path))

            _get_file(hdfs_path, local_path)
        else:
            # TODO: partitioned files

            with temppath() as tpath:
                _temp_dir_path = osp.join(tpath, posixpath.basename(hdfs_path))
                os.makedirs(_temp_dir_path)
                _scrape_dir(hdfs_path, _temp_dir_path)

                if verbose:
                    self.log('Moving {0} to {1}'.format(_temp_dir_path,
                                                        local_path))

                if overwrite and osp.exists(local_path):
                    # swap and delete
                    local_swap_path = util.guid()
                    shutil.move(local_path, local_swap_path)

                    try:
                        shutil.move(_temp_dir_path, local_path)
                        if verbose:
                            msg = 'Deleting original {0}'.format(local_path)
                            self.log(msg)
                        shutil.rmtree(local_swap_path)
                    except:
                        # undo our diddle
                        shutil.move(local_swap_path, local_path)
                else:
                    shutil.move(_temp_dir_path, local_path)

        return dest
