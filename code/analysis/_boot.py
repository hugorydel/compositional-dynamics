"""Put `code/` on the import path, so an analysis can be run from anywhere.

Same arrangement as `tests/_boot.py`: the scripts here live one directory
below the code they read, so `code/` is not on `sys.path` when one is run as a
script.  Import this first, then `_paths`, `relspec` and the figure modules
import exactly as they do upstairs.
"""

import os
import sys

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if CODE not in sys.path:
    sys.path.insert(0, CODE)
