import json
import csv
from pathlib import Path
from tools.dataset.finalize_dataset import (
    load_metadata, split_train_val, write_csv, generate_missing_tags_report,
)

def _write_metadata(path: Path, records: list[dict]) -> None:
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

def test_load_metadata(tmp_path: Path) -> None:
    meta = tmp_path / "metadata.jsonl"
    _write_metadata(meta, [
        {"scene_id": "1", "image_names": ["s1_f0001.jpg", "s1_f0050.jpg"],
         "caption": "A scene.", "tags": ["PAWG"], "missing_tags": []},
    ])
    records = load_metadata(meta)
    assert len(records) == 1
    assert records[0]["scene_id"] == "1"

def test_split_train_val_is_scene_level() -> None:
    records = [
        {"scene_id": str(i), "image_names": [f"s{i}_f001.jpg", f"s{i}_f002.jpg"],
         "caption": f"Caption {i}.", "tags": [], "missing_tags": []}
        for i in range(10)
    ]
    train, val = split_train_val(records, val_fraction=0.1)
    train_ids = {r["scene_id"] for r in train}
    val_ids = {r["scene_id"] for r in val}
    assert not (train_ids & val_ids)
    assert len(val_ids) == 1
    assert len(train_ids) == 9

def test_write_csv(tmp_path: Path) -> None:
    records = [
        {"scene_id": "1", "image_names": ["s1_f0001.jpg"],
         "caption": "A scene.", "tags": [], "missing_tags": []},
    ]
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    csv_path = tmp_path / "train.csv"
    write_csv(records, csv_path, images_dir)
    rows = list(csv.DictReader(csv_path.open()))
    assert len(rows) == 1
    assert rows[0]["caption"] == "A scene."
    assert "s1_f0001.jpg" in rows[0]["filepath"]

def test_missing_tags_report(tmp_path: Path) -> None:
    records = [
        {"scene_id": "1", "missing_tags": ["Blowjob", "POV"], "tags": []},
        {"scene_id": "2", "missing_tags": ["Blowjob"], "tags": []},
        {"scene_id": "3", "missing_tags": [], "tags": []},
    ]
    report = generate_missing_tags_report(records)
    assert "Blowjob" in report
    assert "2" in report
