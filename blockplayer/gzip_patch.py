import gzip


def __enter__(self):
    if self.fileobj is None:
        raise ValueError("I/O operation on closed GzipFile object")
    return self
gzip.GzipFile.__enter__ = __enter__


def __exit__(self, *args):
    self.close()
gzip.GzipFile.__exit__ = __exit__
