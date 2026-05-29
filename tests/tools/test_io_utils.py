import json
from pathlib import Path
from tools.dataset.io_utils import (
    copy_frames_to_dataset, write_caption_files, append_metadata_record,
    dataset_image_name,
)

def test_dataset_image_name() -> None:
    p = Path("/some/dir/scene_19/frame_0001.jpg")
    assert dataset_image_name("19", p) == "s19_f0001.jpg"

def test_copy_frames_creates_files(tmp_path: Path) -> None:
    src = tmp_path / "frames" / "scene_42"
    src.mkdir(parents=True)
    frame = src / "frame_0010.jpg"
    frame.write_bytes(b"fake jpeg data")
    dest = tmp_path / "images"
    dest.mkdir()
    copied = copy_frames_to_dataset("42", [frame], dest)
    assert len(copied) == 1
    dest_file = dest / "s42_f0010.jpg"
    assert dest_file.exists()
    assert dest_file.read_bytes() == b"fake jpeg data"
    assert frame.exists()  # Source NOT deleted

def test_write_caption_files(tmp_path: Path) -> None:
    dest = tmp_path / "images"
    dest.mkdir()
    image_names = ["s19_f0001.jpg", "s19_f0050.jpg"]
    for name in image_names:
        (dest / name).write_bytes(b"img")
    caption = "An amateur scene featuring a PAWG performer with big ass."
    write_caption_files(image_names, caption, dest)
    for name in image_names:
        txt = dest / name.replace(".jpg", ".txt")
        assert txt.exists()
        assert txt.read_text() == caption

def test_append_metadata_record(tmp_path: Path) -> None:
    jsonl = tmp_path / "metadata.jsonl"
    jsonl.touch()
    append_metadata_record(
        jsonl_path=jsonl, scene_id="19", tags=["PAWG", "Big Ass"],
        caption="A non-nude scene.", visual_notes="Performer is dancing in lingerie.",
        missing_tags=["Striptease"], image_names=["s19_f0001.jpg"],
    )
    lines = jsonl.read_text().strip().split("\n")
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["scene_id"] == "19"
    assert record["caption"] == "A non-nude scene."
    assert "Striptease" in record["missing_tags"]
