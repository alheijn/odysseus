import os
import ssl
import sys

import certifi

_original_create_default_context = ssl.create_default_context


def _patched_create_default_context(*args, **kwargs):
    kwargs.setdefault('cafile', certifi.where())
    return _original_create_default_context(*args, **kwargs)


# On some Windows setups, Python 3.11/OpenSSL fails while loading the Windows
# certificate store. Point Jupyter and any HTTPS clients at certifi instead.
os.environ.setdefault('SSL_CERT_FILE', certifi.where())
os.environ.setdefault('REQUESTS_CA_BUNDLE', certifi.where())
ssl.create_default_context = _patched_create_default_context
ssl._create_default_https_context = _patched_create_default_context

from notebook.app import main

sys.argv = [
    'jupyter-notebook',
    '--no-browser',
    '--ip=127.0.0.1',
    '--NotebookApp.token=',
    '--NotebookApp.password=',
    '--port=8888',
]
main()
