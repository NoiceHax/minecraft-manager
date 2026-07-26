"""The container platform boundary.

``import docker`` appears in exactly two modules of this package - ``docker_runtime.py`` and
``streams.py`` - and nowhere else in the project. Everything above sees ``ContainerRuntime`` and
the DTOs.
"""

from __future__ import annotations
