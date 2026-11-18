import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from uie import ruod

ROOT = Path(__file__).resolve().parents[1]


def test_yolo_lines_normalises_and_clips():
    lines = ruod.yolo_lines([(3, 10, 20, 30, 40), (0, -10, 90, 30, 20), (1, 200, 0, 5, 5)], w=100, h=100)
    assert lines[0] == "3 0.250000 0.400000 0.300000 0.400000"
    # clipped at the left and bottom edges: x 0..20, y 90..100
    assert lines[1] == "0 0.100000 0.950000 0.200000 0.100000"
    assert len(lines) == 2  # fully outside: dropped


def _coco(path, files, sizes):
    images = [{"id": i + 1, "file_name": f, "width": w, "height": h} for i, (f, (w, h)) in enumerate(zip(files, sizes))]
    anns = [{"id": i + 1, "image_id": i + 1, "bbox": [w / 4, h / 4, w / 2, h / 2], "category_id": 2, "iscrowd": 0}
            for i, (w, h) in enumerate(sizes)]
    cats = [{"id": 2, "name": "echinus"}, {"id": 1, "name": "holothurian"}]
    path.write_text(json.dumps({"images": images, "annotations": anns, "categories": cats}))


def test_build_set_links_resizes_and_strips_exif(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    Image.new("RGB", (400, 300), (10, 20, 30)).save(src / "small.jpg")
    Image.new("RGB", (3840, 2160), (40, 50, 60)).save(src / "big.jpg")
    exif = Image.Exif()
    exif[0x0112] = 6  # "rotate 90" flag
    Image.new("RGB", (400, 300)).save(src / "rot.jpg", exif=exif)
    files, sizes = ["small.jpg", "big.jpg", "rot.jpg"], [(400, 300), (3840, 2160), (400, 300)]
    _coco(tmp_path / "ann.json", files, sizes)
    names, images = ruod.load_coco(tmp_path / "ann.json")
    assert names == ["holothurian", "echinus"]  # category-id order -> class index

    out = tmp_path / "yolo"
    st = ruod.build_set(out, "train", src, files, images, max_side=1024, workers=0)
    assert st == {"images": 3, "boxes": 3, "link": 1, "resized": 1, "copy": 1}
    assert (out / "images/train/small.jpg").is_symlink()
    with Image.open(out / "images/train/big.jpg") as im:
        assert im.size == (1024, 576)
    with Image.open(out / "images/train/rot.jpg") as im:
        assert im.getexif().get(0x0112, 1) == 1 and im.size == (400, 300)
    # class 1 (echinus), centred box covering half of each side
    assert (out / "labels/train/big.txt").read_text() == "1 0.500000 0.500000 0.500000 0.500000\n"


def test_build_set_reports_size_mismatch(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    Image.new("RGB", (200, 100)).save(src / "a.jpg")
    _coco(tmp_path / "ann.json", ["a.jpg"], [(400, 300)])
    _, images = ruod.load_coco(tmp_path / "ann.json")
    st = ruod.build_set(tmp_path / "out", "test", src, ["a.jpg"], images, workers=0)
    assert st["size_mismatch"] == 1


def test_eval_matches_by_stem_and_counts_resized(tmp_path):
    sys.path.insert(0, str(ROOT))
    from scripts.eval_detection import build
    enh = tmp_path / "enhanced"
    enh.mkdir()
    Image.fromarray(np.zeros((256, 256, 3), np.uint8)).save(enh / "a.png")  # enhancer changed size + extension
    Image.new("RGB", (400, 300)).save(enh / "b.jpg")
    _coco(tmp_path / "ann.json", ["a.jpg", "b.jpg"], [(400, 300), (400, 300)])
    names, images = ruod.load_coco(tmp_path / "ann.json")
    info = build(enh, ["a.jpg", "b.jpg"], images, names, tmp_path / "work")
    assert info == {"images": 2, "resized": 1}
    assert (tmp_path / "work/labels/eval/a.txt").read_text().startswith("1 0.500000")
    assert "val: images/eval" in (tmp_path / "work/data.yaml").read_text()
    with pytest.raises(SystemExit):
        build(enh, ["a.jpg", "zzz.jpg"], images, names, tmp_path / "work2")


def test_bench_split_is_inside_test_and_balanced():
    split = lambda n: (ROOT / "splits" / n).read_text().split()
    bench, test = split("ruod_bench400.txt"), split("ruod_test.txt")
    assert len(bench) == 400 == len(set(bench)) and set(bench) <= set(test)
    assert not set(split("ruod_val.txt")) & set(split("ruod_train_fit.txt"))
    stats = json.loads((ROOT / "splits/ruod_bench400_stats.json").read_text())
    for c in stats["classes"].values():
        assert abs(c["image_share"] - c["image_share_in_test"]) <= 0.015
