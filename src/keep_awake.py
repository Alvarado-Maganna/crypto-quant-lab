"""Keep Windows from going to sleep while a long job runs (no effect on other systems).

Windows API ``SetThreadExecutionState`` (kernel32): calling it with
``ES_CONTINUOUS | ES_SYSTEM_REQUIRED`` tells Windows "the system is in use, do not sleep on
idle" until the same thread calls it again with ``ES_CONTINUOUS`` alone. The screen may still
turn off (that would need ``ES_DISPLAY_REQUIRED``), which does not stop the job.

What it does NOT prevent: closing the laptop lid (if the lid action is "sleep"), pressing the
power button, choosing Sleep in the Start menu, or the battery running out.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


def _set_state(flags):
    import ctypes

    previous = ctypes.windll.kernel32.SetThreadExecutionState(ctypes.c_uint(flags))
    if previous == 0:  # the API returns 0 on failure
        raise OSError("SetThreadExecutionState failed")
    return previous


@contextmanager
def keep_awake(log=print, platform=None, set_state=None):
    """Inside the ``with`` block, Windows does not sleep on idle; restored on exit or Ctrl+C.

    ``platform`` / ``set_state`` exist for tests. On a system other than Windows it only logs.
    """
    platform = sys.platform if platform is None else platform
    if platform != "win32":
        log(f"keep-awake: not Windows ({platform}), nothing to do")
        yield False
        return
    set_state = set_state or _set_state
    try:
        set_state(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    except Exception as error:  # never stop the job because of this
        log(f"keep-awake: could not set it ({type(error).__name__}: {error}); "
            f"set the power plan to never sleep instead")
        yield False
        return
    log("keep-awake: on (Windows will not sleep on idle while this runs)")
    try:
        yield True
    finally:
        try:
            set_state(ES_CONTINUOUS)
            log("keep-awake: off (normal sleep settings restored)")
        except Exception as error:
            log(f"keep-awake: could not restore ({type(error).__name__}: {error})")
