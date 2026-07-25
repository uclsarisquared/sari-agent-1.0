"""Live watching for a distributed benchmark battery.

* ``scan``   - battery output dir -> a view of every attempt (pure filesystem reads)
* ``health`` - collapse detection over an attempt's step records (pure)
* ``server`` - the dashboard: JSON API, latest-frame images, kill endpoint
* ``notify`` - Discord webhooks, driven off the poll loop

Nothing here writes to a run dir except the ``killed_by`` stamp, and nothing here talks to a
sandbox. A watcher crash can never disturb a battery that is hours in.
"""

__all__ = ["health", "scan", "server", "notify"]
