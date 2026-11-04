import io
import json
import tarfile

import pytest

from uie import runstate


def test_pack_roundtrip(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(), b.mkdir()
    (a / "1.png").write_bytes(b"one")
    (b / "2.png").write_bytes(b"two")
    (a / "sub").mkdir()  # directories are skipped
    pack = tmp_path / "x.pack"
    assert runstate.write_pack({"bank": a, "val/LA": b}, pack) == 2
    assert not pack.with_name("x.pack.tmp").exists()

    out = tmp_path / "out"
    assert runstate.read_pack(pack, out) == 2
    assert (out / "bank/1.png").read_bytes() == b"one"
    assert (out / "val/LA/2.png").read_bytes() == b"two"


def test_pack_follows_symlinks(tmp_path):
    (tmp_path / "real.png").write_bytes(b"data")
    d = tmp_path / "d"
    d.mkdir()
    (d / "l.png").symlink_to(tmp_path / "real.png")
    runstate.write_pack({"p": d}, tmp_path / "x.pack")
    runstate.read_pack(tmp_path / "x.pack", tmp_path / "out")
    out = tmp_path / "out/p/l.png"
    assert not out.is_symlink() and out.read_bytes() == b"data"


def test_read_pack_keeps_existing_without_overwrite(tmp_path):
    d = tmp_path / "d"
    d.mkdir()
    (d / "f").write_bytes(b"new")
    runstate.write_pack({"p": d}, tmp_path / "x.pack")
    (tmp_path / "out/p").mkdir(parents=True)
    (tmp_path / "out/p/f").write_bytes(b"old")
    assert runstate.read_pack(tmp_path / "x.pack", tmp_path / "out", overwrite=False) == 0
    assert (tmp_path / "out/p/f").read_bytes() == b"old"


@pytest.mark.parametrize("name", ["../evil", "/abs/evil"])
def test_read_pack_rejects_escaping_names(tmp_path, name):
    pack = tmp_path / "bad.pack"
    with tarfile.open(pack, "w") as t:
        info = tarfile.TarInfo(name)
        info.size = 1
        t.addfile(info, io.BytesIO(b"x"))
    with pytest.raises(ValueError):
        runstate.read_pack(pack, tmp_path / "out")


def test_memlogger_writes_json_lines(tmp_path):
    log = runstate.MemLogger(tmp_path / "mem.log", interval=60, status=lambda: {"epoch": 3})
    log.log(event="x")
    rec = json.loads((tmp_path / "mem.log").read_text().splitlines()[-1])
    assert rec["epoch"] == 3 and rec["event"] == "x"
    for k in ("rss_main", "rss_children", "cg_current", "cg_max", "oom_kills", "host_avail"):
        assert k in rec  # None where the platform has no /proc or cgroup


def test_memlogger_survives_bad_status(tmp_path):
    def boom():
        raise RuntimeError
    rec = runstate.MemLogger(tmp_path / "mem.log", status=boom).log()
    assert "time" in rec


def test_memlogger_thread_stops(tmp_path):
    log = runstate.MemLogger(tmp_path / "mem.log", interval=0.01)
    log.start()
    log.stop()
    log.join(timeout=2)
    assert not log.is_alive()
