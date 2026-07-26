"""On-disk state: the resume marker, the idle deadline, and archived session records.

Load-bearing, not a nicety: the docker log driver is a ~30MB ring, so a long session genuinely
cannot be reconstructed after the fact.
"""

from __future__ import annotations
