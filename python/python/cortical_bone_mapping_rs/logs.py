"""Logging.

Every stage of the Rust library logs through Python's :mod:`logging`, under
the logger ``cortical_bone_mapping`` with one child per Rust module (e.g.
``cortical_bone_mapping.fit.volumetric``); this package's own steps log
under ``cortical_bone_mapping_rs``. Nothing is shown unless logging is
configured:

======== =====================================================================
level    what
======== =====================================================================
INFO     one line per call (fit, refinement, segmentation step) and its outcome
DEBUG    every stage when it starts and ends (sizes, duration), every solver
         iteration, every refinement move, the options in use
TRACE    (level 5) details within stages, e.g. every conjugate-gradient solve
WARNING  a fit that did not converge, a fallback
======== =====================================================================

A stage logs when it starts, so the log of a run that stalls shows where it
is. Set levels with the :mod:`logging` module as usual, or with
:func:`log_to`; a change takes effect at the next call into the library.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional, Union

RUST_LOGGER = "cortical_bone_mapping"
PYTHON_LOGGER = "cortical_bone_mapping_rs"
TRACE = 5
logging.addLevelName(TRACE, "TRACE")

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def log_to(path: Optional[Union[str, Path]] = None, level: Union[int, str] = "DEBUG") -> logging.Handler:
    """Send the library's log records at ``level`` and above to the file
    ``path`` (appended to), or to stderr if ``None``. Returns the handler
    (remove it with ``logging.getLogger(name).removeHandler`` for both
    :data:`RUST_LOGGER` and :data:`PYTHON_LOGGER`).

    >>> import cortical_bone_mapping_rs as cbm
    >>> handler = cbm.log_to("run.log", level="DEBUG")  # doctest: +SKIP
    """
    lvl = TRACE if isinstance(level, str) and level.upper() == "TRACE" else level
    handler: logging.Handler = (logging.FileHandler(path, encoding="utf-8") if path is not None
                                else logging.StreamHandler(sys.stderr))
    handler.setFormatter(logging.Formatter(_FORMAT))
    handler.setLevel(lvl)
    for name in (RUST_LOGGER, PYTHON_LOGGER):
        logger = logging.getLogger(name)
        logger.setLevel(lvl)
        logger.addHandler(handler)
    return handler
