"""Distributed Sari Bench: run a prompt battery across a fleet of Sari Sandbox instances.

Three pieces:

* ``coordinator`` - the sandbox registry. Sims connect to it on startup, report the port they
  self-assigned, and sit in a pool. Benchmark workers lease a sandbox out of that pool and hand it
  back when their attempt finishes.
* ``runner`` - reads a prompt list, expands it to (prompt x attempt), and drives one agent
  subprocess per attempt against a leased sandbox.
* ``protocol`` - the wire format both halves speak, mirrored on the Unity side by
  ``BenchCoordinatorClient.cs``.

Plus, on the runner's machine, three read-only tools over the artefacts a battery leaves behind:
``watch`` (live dashboard with collapse detection and Discord alerts), ``report`` (CSVs), and
``video`` (screenshot replays).
"""

from sari_bench.protocol import SCHEMA_VERSION

__all__ = ["SCHEMA_VERSION"]
