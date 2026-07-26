"""``python -m mcmanager``, equivalent to the ``mcmanager`` console script."""

from __future__ import annotations

import sys

from mcmanager.cli.main import main

if __name__ == "__main__":
    sys.exit(main())
