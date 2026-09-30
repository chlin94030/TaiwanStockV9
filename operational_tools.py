"""Availability/security helpers, intentionally independent of return estimates.

No background work, no orders, no network side effects at import time.
The local scan lock is not a distributed lock or an authorization boundary.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
import errno
import importlib.metadata
import json
import os
import platform
import re
import sys

OPERATIONS_VERSION = "8.2.1-availability-verification"


def safe_error_text(value: Any, limit: int = 240, secrets: Iterable[str] = ()) -> str:
    """Redact credentials BEFORE truncation, including HTTP exceptions with URLs.

    Not a general-purpose data-loss-prevention system. Do not log requests or
    headers containing secrets; known provider secrets are passed explicitly.
    """
    text = str(value)
    for secret in sorted({s for s in secrets if isinstance(s, str) and s}, key=len, reverse=True):
        text = text.replace(secret, "[REDACTED]")
    # Redact bearer values first; otherwise key redaction may remove the word Bearer.
    text = re.sub(r"(?i)(\bBearer\s+)[A-Za-z0-9._~+/-]+=*", r"\1[REDACTED]", text)
    # Handles unquoted and JSON/dict-style key/value representations.
    text = re.sub(
        r"(?i)(\b(?:token|access_token|refresh_token|api[_-]?key|authorization|password|secret)\b"
        r"[\s\"']*[:=][\s\"']*)([^\s&\"'<>,}]+)",
        r"\1[REDACTED]", text,
    )
    # URL user:password credentials are not retained in exported diagnostics.
    text = re.sub(r"(https?://)[^\s/@]+:[^\s/@]+@", r"\1[REDACTED]@", text, flags=re.I)
    return text[:max(0, int(limit))]


class ScanBusyError(RuntimeError):
    """Another cooperating process/session owns the heavy-scan lock."""


@contextmanager
def exclusive_scan(data_dir: Path):
    """Nonblocking OS lock for one heavy scan per local data directory.

    The lock file is never unlinked: doing that can split concurrent waiters
    across different inodes. The OS releases the advisory lock after process
    exit; the file's metadata is diagnostic only, not a stale-lock TTL.
    Doctor/cache reads do not acquire this lock. Separate machines/data dirs
    need their own coordination and memory budgets.
    """
    root = Path(data_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    path = root / ".market_scan.lock"
    handle = path.open("a+b")
    locked = False
    unlock = None
    try:
        if os.name == "posix":
            import fcntl
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                if exc.errno in (errno.EACCES, errno.EAGAIN):
                    raise ScanBusyError("A market scan is already running for this data directory") from exc
                raise
            unlock = lambda: fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        elif os.name == "nt":
            import msvcrt
            if path.stat().st_size == 0:
                handle.write(b"\0"); handle.flush()
            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                if exc.errno in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                    raise ScanBusyError("A market scan is already running for this data directory") from exc
                raise
            def unlock_windows():
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            unlock = unlock_windows
        else:
            raise RuntimeError("Safe local scan coordination unsupported on this platform")
        locked = True
        # Fixed-length metadata is replaced only after acquiring the lock.
        meta = json.dumps({"pid": os.getpid(), "started_utc": datetime.now(timezone.utc).isoformat(),
                           "operations_version": OPERATIONS_VERSION}, separators=(",", ":")).encode()
        handle.seek(0); handle.truncate(); handle.write(meta); handle.flush()
        yield
    finally:
        try:
            if locked and unlock is not None:
                unlock()
        finally:
            handle.close()


def runtime_inventory() -> dict:
    """Offline metadata only: package installed != import/network health verified."""
    packages = {}
    for name in ("streamlit", "pandas", "numpy", "plotly", "requests", "yfinance", "curl_cffi", "websockets"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "operations_version": OPERATIONS_VERSION,
        "python": platform.python_version(),
        "platform": sys.platform,
        "machine": platform.machine(),
        "installed_packages": packages,
        "live_connectivity_tested": False,
        "note": "Installed versions and backend load state are not proof of provider connectivity or profitability",
    }
