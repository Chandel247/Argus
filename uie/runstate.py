"""Helpers for training that spans several Kaggle sessions.

Packs: a directory tree saved as one uncompressed tar, written to a temp file
and renamed. Kaggle's output API downloads files one by one (800 small bank
images made fetches slow and flaky), and Kaggle unpacks .zip/.tar uploads in
datasets, so packs use the extension .pack.

Memory telemetry: host RAM is the suspected cause of a SIGKILL with no Python
traceback, and nothing recorded it. MemLogger appends one JSON line per
interval with the process-tree RSS and the container's cgroup counters
(including the OOM-kill count), flushed per line so it survives a kill.
"""
from __future__ import annotations

import json
import os
import tarfile
import threading
import time
from pathlib import Path, PurePosixPath


# ---------------------------------------------------------------- packs

def write_pack(dirs: dict[str, Path], dest: Path) -> int:
    """Pack the regular files of each dir in `dirs` ({arcname_prefix: dir})
    into `dest` atomically. Returns the number of files."""
    dest = Path(dest)
    tmp = dest.with_name(dest.name + ".tmp")
    n = 0
    with tarfile.open(tmp, "w") as t:
        for prefix, d in dirs.items():
            for p in sorted(Path(d).iterdir()):
                if p.is_file():  # symlinks are followed: the pack holds content
                    real = p.resolve()
                    info = t.gettarinfo(str(real), arcname=f"{prefix}/{p.name}")
                    with open(real, "rb") as f:
                        t.addfile(info, f)
                    n += 1
    os.replace(tmp, dest)
    return n


def read_pack(src: Path, root: Path, overwrite: bool = True) -> int:
    """Extract a pack under `root`. Only plain files with relative,
    non-escaping names are accepted. Returns the number of files."""
    root = Path(root)
    n = 0
    with tarfile.open(src, "r") as t:
        for m in t.getmembers():
            name = PurePosixPath(m.name)
            if not m.isfile() or name.is_absolute() or ".." in name.parts:
                raise ValueError(f"unsafe member in {src}: {m.name!r}")
            out = root.joinpath(*name.parts)
            if out.exists() and not overwrite:
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.with_name(out.name + ".tmp")
            with t.extractfile(m) as f, open(tmp, "wb") as g:
                g.write(f.read())
            os.replace(tmp, out)
            n += 1
    return n


# ---------------------------------------------------------------- memory

def _read(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def _num(s):
    return int(s) if s and s.isdigit() else None


def cgroup_memory() -> dict:
    """Container memory counters (cgroup v2, falling back to v1). Values in
    bytes; None when unavailable (e.g. not on Linux)."""
    v2 = Path("/sys/fs/cgroup")
    v1 = v2 / "memory"
    cur = _num(_read(v2 / "memory.current")) or _num(_read(v1 / "memory.usage_in_bytes"))
    lim = _num(_read(v2 / "memory.max")) or _num(_read(v1 / "memory.limit_in_bytes"))
    peak = _num(_read(v2 / "memory.peak")) or _num(_read(v1 / "memory.max_usage_in_bytes"))
    oom = None
    events = _read(v2 / "memory.events") or _read(v1 / "memory.oom_control")
    for line in (events or "").splitlines():
        k, _, v = line.partition(" ")
        if k == "oom_kill":
            oom = _num(v)
    return {"cg_current": cur, "cg_max": lim, "cg_peak": peak, "oom_kills": oom}


def host_meminfo() -> dict:
    info = {}
    for line in (_read("/proc/meminfo") or "").splitlines():
        k, _, v = line.partition(":")
        if k in ("MemTotal", "MemAvailable"):
            info[k] = int(v.split()[0]) * 1024
    return {"host_total": info.get("MemTotal"), "host_avail": info.get("MemAvailable")}


def tree_rss(pid: int | None = None) -> dict:
    """RSS of `pid` and of all its descendants (DataLoader workers), bytes."""
    pid = pid or os.getpid()
    proc = Path("/proc")
    if not proc.is_dir():
        return {"rss_main": None, "rss_children": None, "n_children": None}
    parent = {}
    for p in proc.iterdir():
        if p.name.isdigit():
            stat = _read(p / "stat")
            if stat:  # "pid (comm) state ppid ..."; comm may contain spaces
                parent[int(p.name)] = int(stat.rsplit(")", 1)[1].split()[1])
    kids, frontier = [], [pid]
    while frontier:
        cur = frontier.pop()
        for c, pp in parent.items():
            if pp == cur:
                kids.append(c)
                frontier.append(c)

    def rss(p):
        for line in (_read(proc / str(p) / "status") or "").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
        return 0

    return {"rss_main": rss(pid), "rss_children": sum(rss(c) for c in kids), "n_children": len(kids)}


def memory_snapshot() -> dict:
    return {**tree_rss(), **cgroup_memory(), **host_meminfo()}


def gb(x):
    return None if x is None else round(x / 2**30, 2)


class MemLogger(threading.Thread):
    """Background thread appending memory snapshots to `path` every
    `interval` seconds. `status` returns extra fields (e.g. epoch, iteration)."""

    def __init__(self, path: Path, interval: float = 60, status=None):
        super().__init__(daemon=True, name="memlogger")
        self.path, self.interval, self.status = Path(path), interval, status or dict
        self._halt = threading.Event()

    def log(self, **extra):
        rec = {"time": time.strftime("%Y-%m-%dT%H:%M:%S")}
        try:
            rec.update(self.status())
        except Exception:  # never let telemetry break training
            pass
        rec.update({k: gb(v) if k not in ("n_children", "oom_kills") else v
                    for k, v in memory_snapshot().items()})
        rec.update(extra)
        with open(self.path, "a", buffering=1) as f:
            f.write(json.dumps(rec) + "\n")
        return rec

    def run(self):
        while not self._halt.wait(self.interval):
            try:
                self.log()
            except Exception as e:  # pragma: no cover
                print(f"memlogger: {e}", flush=True)

    def stop(self):
        self._halt.set()
