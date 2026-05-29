"""Tests for batch API module."""
import base64
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from tools.dataset.batch_api import (
    BatchJob,
    CollectStats,
    check_scene_completion,
    collect_batch_results,
    load_batch_state,
    poll_batch_jobs,
    prepare_batch_chunks,
    save_batch_state,
    submit_batch_job,
    update_caption_progress,
    upload_jsonl,
)


def test_batch_job_round_trip(tmp_path: Path) -> None:
    """BatchJob serializes to/from JSON via state file."""
    state_file = tmp_path / "batch_state.json"
    job = BatchJob(
        name="batches/abc123",
        display_name="chunk-001",
        state="BATCH_STATE_PENDING",
        submitted_at="2026-02-25T10:00:00Z",
        frame_count=25000,
        scene_ids=[1, 2, 3],
        file_name="files/xyz",
        result_file=None,
        collected=False,
        stats=None,
    )
    save_batch_state(state_file, [job])
    loaded = load_batch_state(state_file)
    assert len(loaded) == 1
    assert loaded[0].name == "batches/abc123"
    assert loaded[0].frame_count == 25000
    assert loaded[0].collected is False


def test_batch_state_empty_file(tmp_path: Path) -> None:
    """load_batch_state returns empty list for missing file."""
    state_file = tmp_path / "batch_state.json"
    assert load_batch_state(state_file) == []


def test_collect_stats_serialization() -> None:
    """CollectStats round-trips through dict."""
    stats = CollectStats(captions_written=100, errors=5, scenes_completed=10)
    d = stats.to_dict()
    loaded = CollectStats.from_dict(d)
    assert loaded.captions_written == 100
    assert loaded.errors == 5


def test_prepare_batch_chunks_creates_jsonl(tmp_path: Path) -> None:
    """prepare_batch_chunks creates JSONL files with correct structure."""
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    scene_dir = frames_dir / "scene_1"
    scene_dir.mkdir()
    batch_dir = tmp_path / "batch_jobs"

    for i in range(3):
        frame = scene_dir / f"frame_{i+1:04d}.jpg"
        frame.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 10)

    mock_selection = MagicMock()
    mock_selection.path = str(scene_dir / "frame_0001.jpg")
    mock_selection2 = MagicMock()
    mock_selection2.path = str(scene_dir / "frame_0002.jpg")

    with patch("tools.dataset.batch_api.select_frames_for_scene") as mock_select, \
         patch("tools.dataset.batch_api.dataset_image_name") as mock_name:
        mock_select.return_value = [mock_selection, mock_selection2]
        mock_name.side_effect = lambda sid, fp: f"s{sid}_f{Path(fp).stem.replace('frame_', '')}.jpg"

        chunks = prepare_batch_chunks(
            index=MagicMock(scene_id_list=[1]),
            completed_scenes=set(),
            images_dir=images_dir,
            frames_dir=frames_dir,
            batch_dir=batch_dir,
            max_frames=20,
            chunk_size=100,
        )

    assert len(chunks) == 1
    assert chunks[0].exists()
    lines = chunks[0].read_text().strip().split("\n")
    assert len(lines) == 2
    for line in lines:
        record = json.loads(line)
        assert "key" in record
        assert "request" in record
        assert "contents" in record["request"]
        parts = record["request"]["contents"][0]["parts"]
        assert parts[0]["inlineData"]["mimeType"] == "image/jpeg"
        assert "text" in parts[1]


def test_upload_jsonl_calls_file_api(tmp_path: Path) -> None:
    """upload_jsonl uploads file via resumable upload protocol."""
    jsonl = tmp_path / "test.jsonl"
    jsonl.write_text('{"key":"r1","request":{}}\n')

    with patch("tools.dataset.batch_api.requests") as mock_req:
        init_resp = MagicMock()
        init_resp.status_code = 200
        init_resp.headers = {"X-Goog-Upload-URL": "https://upload.example.com/resume"}

        upload_resp = MagicMock()
        upload_resp.status_code = 200
        upload_resp.json.return_value = {"file": {"name": "files/abc123"}}

        mock_req.post.side_effect = [init_resp, upload_resp]

        file_name = upload_jsonl(jsonl, "test-api-key")

    assert file_name == "files/abc123"
    assert mock_req.post.call_count == 2


def test_submit_batch_job_calls_api(tmp_path: Path) -> None:
    """submit_batch_job POSTs to batchGenerateContent."""
    with patch("tools.dataset.batch_api.requests") as mock_req:
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "name": "batches/xyz789",
            "metadata": {"state": "BATCH_STATE_PENDING"},
        }
        mock_req.post.return_value = resp

        job = submit_batch_job(
            file_name="files/abc123",
            model="gemini-3-flash-preview",
            api_key="test-key",
            display_name="chunk-001",
            frame_count=25000,
            scene_ids=[1, 2, 3],
        )

    assert job.name == "batches/xyz789"
    assert job.state == "BATCH_STATE_PENDING"
    assert job.frame_count == 25000


def test_poll_updates_job_state() -> None:
    """poll_batch_jobs updates state from API response."""
    job = BatchJob(
        name="batches/abc", display_name="chunk-001",
        state="BATCH_STATE_PENDING", submitted_at="",
        frame_count=10, scene_ids=[1],
    )

    with patch("tools.dataset.batch_api.requests") as mock_req:
        resp = MagicMock()
        resp.json.return_value = {
            "name": "batches/abc",
            "metadata": {"state": "BATCH_STATE_SUCCEEDED"},
            "response": {"responsesFile": "files/result123"},
        }
        mock_req.get.return_value = resp

        updated = poll_batch_jobs([job], "test-key")

    assert updated[0].state == "BATCH_STATE_SUCCEEDED"
    assert updated[0].result_file == "files/result123"


def test_collect_writes_txt_files(tmp_path: Path) -> None:
    """collect_batch_results parses response JSONL and writes .txt files."""
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    result_content = (
        '{"key": "s1_f0001", "response": {"candidates": [{"content": {"parts": [{"text": "A test caption."}]}, "finishReason": "STOP"}], "usageMetadata": {"promptTokenCount": 100, "candidatesTokenCount": 20}}}\n'
        '{"key": "s1_f0002", "response": {"candidates": [{"content": {"parts": [{"text": "Another caption."}]}, "finishReason": "STOP"}], "usageMetadata": {"promptTokenCount": 100, "candidatesTokenCount": 20}}}\n'
    )

    job = BatchJob(
        name="batches/abc", display_name="chunk-001",
        state="BATCH_STATE_SUCCEEDED", submitted_at="",
        frame_count=2, scene_ids=[1],
        result_file="files/result123",
    )

    with patch("tools.dataset.batch_api.requests") as mock_req:
        resp = MagicMock()
        resp.content = result_content.encode("utf-8")
        mock_req.get.return_value = resp

        result = collect_batch_results(job, images_dir, "test-key")

    assert result.stats is not None
    assert result.stats.captions_written == 2
    assert result.stats.errors == 0
    assert (images_dir / "s1_f0001.txt").read_text() == "A test caption."
    assert (images_dir / "s1_f0002.txt").read_text() == "Another caption."


def test_full_lifecycle_mocked(tmp_path: Path) -> None:
    """Full lifecycle: prepare → submit → poll → collect."""
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    scene_dir = frames_dir / "scene_1"
    scene_dir.mkdir()
    (scene_dir / "frame_0001.jpg").write_bytes(b"\xff\xd8" + b"\x00" * 20)
    batch_dir = tmp_path / "batch_jobs"
    state_file = tmp_path / "batch_state.json"

    mock_sel = MagicMock()
    mock_sel.path = str(scene_dir / "frame_0001.jpg")

    with patch("tools.dataset.batch_api.select_frames_for_scene", return_value=[mock_sel]), \
         patch("tools.dataset.batch_api.dataset_image_name", return_value="s1_f0001.jpg"):
        chunks = prepare_batch_chunks(
            index=MagicMock(scene_id_list=[1]),
            completed_scenes=set(),
            images_dir=images_dir,
            frames_dir=frames_dir,
            batch_dir=batch_dir,
            max_frames=20,
            chunk_size=100,
        )

    assert len(chunks) == 1

    # Mock upload + submit
    with patch("tools.dataset.batch_api.requests") as mock_req:
        init_resp = MagicMock(status_code=200)
        init_resp.headers = {"X-Goog-Upload-URL": "https://example.com/upload"}
        upload_resp = MagicMock(status_code=200)
        upload_resp.json.return_value = {"file": {"name": "files/f1"}}
        submit_resp = MagicMock(status_code=200)
        submit_resp.json.return_value = {
            "name": "batches/b1",
            "metadata": {"state": "BATCH_STATE_PENDING"},
        }
        mock_req.post.side_effect = [init_resp, upload_resp, submit_resp]

        file_name = upload_jsonl(chunks[0], "key")
        job = submit_batch_job(
            file_name=file_name, model="gemini-3-flash-preview",
            api_key="key", display_name="chunk-001",
            frame_count=1, scene_ids=[1],
        )

    assert job.name == "batches/b1"

    # Mock poll → succeeded
    with patch("tools.dataset.batch_api.requests") as mock_req:
        resp = MagicMock()
        resp.json.return_value = {
            "name": "batches/b1",
            "metadata": {"state": "BATCH_STATE_SUCCEEDED"},
            "response": {"responsesFile": "files/r1"},
        }
        mock_req.get.return_value = resp
        poll_batch_jobs([job], "key")

    assert job.state == "BATCH_STATE_SUCCEEDED"
    assert job.result_file == "files/r1"

    # Mock collect
    result_jsonl = '{"key": "s1_f0001", "response": {"candidates": [{"content": {"parts": [{"text": "Test caption"}]}, "finishReason": "STOP"}], "usageMetadata": {}}}\n'
    with patch("tools.dataset.batch_api.requests") as mock_req:
        resp = MagicMock()
        resp.content = result_jsonl.encode()
        mock_req.get.return_value = resp
        collect_batch_results(job, images_dir, "key")

    assert job.collected is True
    assert (images_dir / "s1_f0001.txt").read_text() == "Test caption"

    # Save and reload state
    save_batch_state(state_file, [job])
    loaded = load_batch_state(state_file)
    assert loaded[0].collected is True
    assert loaded[0].stats.captions_written == 1


def test_check_scene_completion_all_done(tmp_path: Path) -> None:
    """check_scene_completion returns scene ID when all frames have .txt files."""
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    scene_dir = frames_dir / "scene_1"
    scene_dir.mkdir()

    # Create the frame files the selector would return
    (scene_dir / "frame_0001.jpg").write_bytes(b"\xff\xd8" + b"\x00" * 10)
    (scene_dir / "frame_0002.jpg").write_bytes(b"\xff\xd8" + b"\x00" * 10)

    # Write caption .txt files (as if batch collection wrote them)
    (images_dir / "s1_f0001.txt").write_text("Caption one.", encoding="utf-8")
    (images_dir / "s1_f0002.txt").write_text("Caption two.", encoding="utf-8")

    mock_sel1 = MagicMock()
    mock_sel1.path = str(scene_dir / "frame_0001.jpg")
    mock_sel2 = MagicMock()
    mock_sel2.path = str(scene_dir / "frame_0002.jpg")

    with patch("tools.dataset.batch_api.select_frames_for_scene", return_value=[mock_sel1, mock_sel2]), \
         patch("tools.dataset.batch_api.dataset_image_name") as mock_name:
        mock_name.side_effect = lambda sid, fp: f"s{sid}_f{Path(fp).stem.replace('frame_', '')}.jpg"

        completed = check_scene_completion(
            scene_ids=[1],
            images_dir=images_dir,
            index=MagicMock(),
            frames_dir=frames_dir,
        )

    assert completed == [1]


def test_check_scene_completion_partial(tmp_path: Path) -> None:
    """check_scene_completion excludes scenes with missing or errored frames."""
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    scene_dir = frames_dir / "scene_1"
    scene_dir.mkdir()

    (scene_dir / "frame_0001.jpg").write_bytes(b"\xff\xd8" + b"\x00" * 10)
    (scene_dir / "frame_0002.jpg").write_bytes(b"\xff\xd8" + b"\x00" * 10)

    # Only one frame has a caption; the other has an error
    (images_dir / "s1_f0001.txt").write_text("Good caption.", encoding="utf-8")
    (images_dir / "s1_f0002.txt").write_text("[ERROR: content filtered]", encoding="utf-8")

    mock_sel1 = MagicMock()
    mock_sel1.path = str(scene_dir / "frame_0001.jpg")
    mock_sel2 = MagicMock()
    mock_sel2.path = str(scene_dir / "frame_0002.jpg")

    with patch("tools.dataset.batch_api.select_frames_for_scene", return_value=[mock_sel1, mock_sel2]), \
         patch("tools.dataset.batch_api.dataset_image_name") as mock_name:
        mock_name.side_effect = lambda sid, fp: f"s{sid}_f{Path(fp).stem.replace('frame_', '')}.jpg"

        completed = check_scene_completion(
            scene_ids=[1],
            images_dir=images_dir,
            index=MagicMock(),
            frames_dir=frames_dir,
        )

    assert completed == []


def test_update_caption_progress_creates_and_merges(tmp_path: Path) -> None:
    """update_caption_progress creates file if missing, and merges on second call."""
    progress_path = tmp_path / "caption_progress.json"

    # First call — creates file
    update_caption_progress(progress_path, [1, 2], captions_written=10, errors=1)
    data = json.loads(progress_path.read_text())
    assert set(data["completed_scenes"]) == {1, 2}
    assert data["total_frames_captioned"] == 10
    assert data["errors"] == 1

    # Second call — merges (scene 2 is deduped, scene 3 is added)
    update_caption_progress(progress_path, [2, 3], captions_written=5, errors=0)
    data = json.loads(progress_path.read_text())
    assert set(data["completed_scenes"]) == {1, 2, 3}
    assert data["total_frames_captioned"] == 15
    assert data["errors"] == 1
