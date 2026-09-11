"""Put `code/` on the import path, so a test can be run from anywhere.

A test lives one directory below the code it exercises, so `code/` is not on
`sys.path` when the test is run as a script.  Importing this first fixes that,
and `_paths`, `relspec` and `style` then import exactly as they do upstairs.

Import it before anything else in the project:

    import _boot  # noqa: F401
    import _paths  # noqa: F401
    from relspec import worlds

The two `noqa` markers are load-bearing: both modules are imported purely for
their effect on `sys.path`, so a linter would otherwise remove them.
"""

import os
import sys

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if CODE not in sys.path:
    sys.path.insert(0, CODE)
