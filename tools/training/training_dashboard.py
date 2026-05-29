#!/usr/bin/env python3
"""LoRA Training Dashboard — live monitoring and control of training runs.

Serves a web dashboard that polls training state files and displays
real-time progress, metrics, GPU stats, and nearest-neighbor galleries.

Usage:
    uv run python tools/training/training_dashboard.py
    uv run python tools/training/training_dashboard.py --port 9874
    uv run python tools/training/training_dashboard.py --runs-dir /path/to/runs
"""
from __future__ import annotations

import argparse
import collections
import json
import logging
import math
import mimetypes
import os
import signal
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from socketserver import ThreadingMixIn
from typing import Any
from urllib.parse import parse_qs, urlparse

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_RUNS_DIR = PROJECT_ROOT / "assets" / "lora_training" / "runs"
HTML_FILE = Path(__file__).resolve().parent / "training_dashboard.html"

TRAINER_MODULE = "tools.training.train_lora"
LOG_BUFFER_SIZE = 500
DEFAULT_CSV = PROJECT_ROOT / "assets" / "lora_dataset" / "train_clean.csv"


# ── Process Management ───────────────────────────────────────────────────


def _find_orphan_process(module_name: str) -> int | None:
    """Find an already-running process for *module_name* not managed by us."""
    my_pid = os.getpid()
    try:
        result = subprocess.run(
            ["pgrep", "-f", module_name],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            for line in result.stdout.strip().splitlines():
                pid = int(line.strip())
                if pid == my_pid:
                    continue
                try:
                    os.kill(pid, 0)
                    return pid
                except OSError:
                    continue
    except (subprocess.TimeoutExpired, ValueError, OSError):
        pass
    return None


class ProcessManager:
    """Manages a subprocess lifecycle with log capture and orphan detection."""

    def __init__(self, module_name: str, display_name: str) -> None:
        self._module = module_name
        self._display = display_name
        self._proc: subprocess.Popen[bytes] | None = None
        self._orphan_pid: int | None = None
        self._log: collections.deque[str] = collections.deque(maxlen=LOG_BUFFER_SIZE)
        self._lock = threading.Lock()
        self._reader_thread: threading.Thread | None = None
        self._adopt_orphan()

    def _adopt_orphan(self) -> None:
        """Detect a process from a previous dashboard session."""
        orphan = _find_orphan_process(self._module)
        if orphan is not None:
            with self._lock:
                self._orphan_pid = orphan
                self._log.append(
                    f"[dashboard] Adopted orphan {self._display} PID {orphan}"
                )

    def _orphan_alive(self) -> bool:
        """Check if the adopted orphan process is still running."""
        if self._orphan_pid is None:
            return False
        try:
            os.kill(self._orphan_pid, 0)
            return True
        except OSError:
            self._orphan_pid = None
            return False

    @property
    def is_running(self) -> bool:
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                return True
            if self._orphan_alive():
                return True
            # Re-check for a process started outside the dashboard
            orphan = _find_orphan_process(self._module)
            if orphan is not None:
                self._orphan_pid = orphan
                self._log.append(
                    f"[dashboard] Detected {self._display} PID {orphan}"
                )
                return True
            return False

    @property
    def pid(self) -> int | None:
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                return self._proc.pid
            if self._orphan_alive():
                return self._orphan_pid
            return None

    @property
    def exit_code(self) -> int | None:
        with self._lock:
            if self._proc is not None:
                return self._proc.poll()
            return None

    def get_log(self, last_n: int = 200) -> list[str]:
        with self._lock:
            lines = list(self._log)
        return lines[-last_n:]

    def launch(self, cmd_args: list[str]) -> dict[str, Any]:
        """Launch the process as a subprocess.

        *cmd_args* are appended after ``[sys.executable, "-m", module]``.
        Raises RuntimeError if already running.
        """
        if self.is_running:
            raise RuntimeError(f"{self._display} is already running")

        cmd = [sys.executable, "-m", self._module] + cmd_args

        with self._lock:
            self._orphan_pid = None
            self._log.clear()
            self._log.append(f"[dashboard] Launching {self._display}: {' '.join(cmd[2:])}")

            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=str(PROJECT_ROOT),
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )

            self._reader_thread = threading.Thread(
                target=self._read_output,
                daemon=True,
            )
            self._reader_thread.start()

        return {"pid": self._proc.pid, "status": "launched"}

    def stop(self) -> dict[str, Any]:
        """Send SIGTERM to the process, escalate to SIGKILL if needed."""
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                pid = self._proc.pid
                self._log.append(f"[dashboard] Sending SIGTERM to {self._display} PID {pid}")
                self._proc.terminate()
            elif self._orphan_pid is not None and self._orphan_alive():
                pid = self._orphan_pid
                self._log.append(
                    f"[dashboard] Sending SIGTERM to orphan {self._display} PID {pid}"
                )
                os.kill(pid, signal.SIGTERM)
                self._orphan_pid = None
                return {"status": "stopped", "pid": pid}
            else:
                return {"status": "not_running"}

        # Wait for graceful shutdown (managed process)
        try:
            self._proc.wait(timeout=5)
            with self._lock:
                self._log.append(
                    f"[dashboard] {self._display} PID {pid} exited "
                    f"(code {self._proc.returncode})"
                )
        except subprocess.TimeoutExpired:
            with self._lock:
                self._log.append(
                    f"[dashboard] {self._display} PID {pid} did not exit, sending SIGKILL"
                )
                self._proc.kill()
                self._proc.wait(timeout=3)
                self._log.append(f"[dashboard] {self._display} PID {pid} killed")

        return {"status": "stopped", "pid": pid}

    def _read_output(self) -> None:
        """Background thread: reads process stdout into the log buffer."""
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        try:
            for raw_line in proc.stdout:
                line = raw_line.decode("utf-8", errors="replace").rstrip("\n\r")
                with self._lock:
                    self._log.append(line)
        except (OSError, ValueError):
            pass
        finally:
            with self._lock:
                if proc.poll() is not None:
                    self._log.append(
                        f"[dashboard] {self._display} exited with code {proc.returncode}"
                    )


training_manager = ProcessManager(TRAINER_MODULE, "Training")


# ── Helpers ──────────────────────────────────────────────────────────────


def _sanitize_floats(obj: Any) -> Any:
    """Replace inf/nan floats with None so JSON serialization is RFC-compliant."""
    if isinstance(obj, float) and (math.isinf(obj) or math.isnan(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize_floats(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_floats(v) for v in obj]
    return obj


def _json_safe(data: Any) -> str:
    """Serialize *data* to JSON, converting inf/nan to null."""
    return json.dumps(_sanitize_floats(data))


def _read_json(path: Path) -> dict[str, Any]:
    """Read a JSON file, returning empty dict if missing or corrupt."""
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file, returning list of dicts. Skips corrupt lines."""
    if not path.exists():
        return []
    entries: list[dict[str, Any]] = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        pass
    return entries


# ── GPU Stats ────────────────────────────────────────────────────────────

_gpu_cache: dict[str, Any] | None = None
_gpu_cache_time: float = 0.0
_GPU_CACHE_TTL = 5.0


def get_gpu_stats() -> dict[str, Any]:
    """Query nvidia-smi for GPU stats. Cached for 5 seconds."""
    global _gpu_cache, _gpu_cache_time

    now = time.time()
    if _gpu_cache is not None and (now - _gpu_cache_time) < _GPU_CACHE_TTL:
        return _gpu_cache

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total,"
                "temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            _gpu_cache = {"available": False}
            _gpu_cache_time = now
            return _gpu_cache

        # Parse first GPU line
        line = result.stdout.strip().splitlines()[0]
        parts = [p.strip() for p in line.split(",")]
        _gpu_cache = {
            "available": True,
            "gpu_util": int(parts[0]),
            "vram_used_mb": int(parts[1]),
            "vram_total_mb": int(parts[2]),
            "temp_c": int(parts[3]),
            "power_w": float(parts[4]),
        }
    except (FileNotFoundError, subprocess.TimeoutExpired, (ValueError, IndexError)):
        _gpu_cache = {"available": False}

    _gpu_cache_time = now
    return _gpu_cache  # type: ignore[return-value]


# ── Run Management ───────────────────────────────────────────────────────


def list_runs(runs_dir: Path) -> list[dict[str, Any]]:
    """Scan for run directories containing config.json.

    Returns list sorted by modification time (newest first).
    Each entry has: name, status, epoch, total_epochs, train_loss, val_loss, created.
    """
    if not runs_dir.is_dir():
        return []

    runs: list[dict[str, Any]] = []
    for entry in runs_dir.iterdir():
        if not entry.is_dir():
            continue
        config_path = entry / "config.json"
        if not config_path.exists():
            continue

        state = _read_json(entry / "train_state.json")
        config = _read_json(config_path)

        runs.append({
            "name": entry.name,
            "status": state.get("status", "unknown"),
            "epoch": state.get("epoch", 0),
            "total_epochs": state.get("total_epochs", config.get("epochs", 0)),
            "step": state.get("step", 0),
            "total_steps": state.get("total_steps", 0),
            "train_loss": state.get("train_loss", 0.0),
            "val_loss": state.get("val_loss", 0.0),
            "best_val_loss": state.get("best_val_loss", 0.0),
            "lr": state.get("lr", 0.0),
            "created": _get_dir_ctime(entry),
            "mtime": entry.stat().st_mtime,
        })

    # Sort newest first
    runs.sort(key=lambda r: r.get("mtime", 0), reverse=True)
    # Remove mtime from output
    for r in runs:
        r.pop("mtime", None)
    return runs


def _get_dir_ctime(path: Path) -> str:
    """Return directory creation time as ISO string."""
    try:
        return time.strftime(
            "%Y-%m-%dT%H:%M:%S",
            time.localtime(path.stat().st_ctime),
        )
    except OSError:
        return ""


def get_run_status(runs_dir: Path, run_name: str) -> dict[str, Any]:
    """Read train_state.json + GPU stats + list eval results for a run."""
    run_dir = runs_dir / run_name
    if not run_dir.is_dir():
        return {"error": f"Run '{run_name}' not found"}

    state = _read_json(run_dir / "train_state.json")
    config = _read_json(run_dir / "config.json")
    gpu = get_gpu_stats()

    # List available eval epochs
    eval_dir = run_dir / "eval"
    eval_epochs: list[int] = []
    if eval_dir.is_dir():
        for f in eval_dir.iterdir():
            if f.suffix == ".json" and f.stem.startswith("epoch_"):
                try:
                    eval_epochs.append(int(f.stem.split("_")[1]))
                except (ValueError, IndexError):
                    pass
    eval_epochs.sort()

    # Count probes if probes.json exists (may be a list or {"probes": [...]})
    probes_data: Any = {}
    if eval_dir.is_dir():
        probes_path = eval_dir / "probes.json"
        if probes_path.exists():
            try:
                probes_data = json.loads(probes_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                probes_data = {}
    if isinstance(probes_data, list):
        probe_count = len(probes_data)
    elif isinstance(probes_data, dict):
        probe_count = len(probes_data.get("probes", []))
    else:
        probe_count = 0

    # Text query count
    text_query_count = 0
    if eval_dir.is_dir():
        tq_path = eval_dir / "text_queries.json"
        if tq_path.exists():
            try:
                tq_data = json.loads(tq_path.read_text(encoding="utf-8"))
                if isinstance(tq_data, list):
                    text_query_count = len(tq_data)
            except (json.JSONDecodeError, OSError):
                pass

    return {
        "run_name": run_name,
        "status": state.get("status", "idle"),
        "epoch": state.get("epoch", 0),
        "total_epochs": state.get("total_epochs", config.get("epochs", 0)),
        "step": state.get("step", 0),
        "total_steps": state.get("total_steps", 0),
        "train_loss": state.get("train_loss", 0.0),
        "val_loss": state.get("val_loss", 0.0),
        "best_val_loss": state.get("best_val_loss", 0.0),
        "lr": state.get("lr", 0.0),
        "elapsed_seconds": state.get("elapsed_seconds", 0.0),
        "eta_seconds": state.get("eta_seconds", 0.0),
        "timestamp": state.get("timestamp", ""),
        "gpu": gpu,
        "eval_epochs": eval_epochs,
        "probe_count": probe_count,
        "text_query_count": text_query_count,
        "config": config,
    }


def get_loss_history(runs_dir: Path, run_name: str) -> list[dict[str, Any]]:
    """Read train_log.jsonl for a run, return as list of dicts."""
    log_path = runs_dir / run_name / "train_log.jsonl"
    return _read_jsonl(log_path)


def get_eval_metrics(runs_dir: Path, run_name: str, epoch: int) -> dict[str, Any]:
    """Read eval metrics for a specific epoch."""
    eval_path = runs_dir / run_name / "eval" / f"epoch_{epoch:03d}.json"
    return _read_json(eval_path)


# ── HTTP Server ──────────────────────────────────────────────────────────


class ThreadedServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    runs_dir: Path = DEFAULT_RUNS_DIR


class DashboardHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the training dashboard."""

    # Set by main() before server starts
    runs_dir: Path = DEFAULT_RUNS_DIR

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        if path in ("/", ""):
            self._serve_html()
        elif path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
        elif path == "/api/status":
            run = query.get("run", [""])[0]
            if run:
                self._send_json(get_run_status(self.runs_dir, run))
            else:
                # Return status for newest run, or empty
                runs = list_runs(self.runs_dir)
                if runs:
                    self._send_json(get_run_status(self.runs_dir, runs[0]["name"]))
                else:
                    self._send_json({"status": "idle", "run_name": ""})
        elif path == "/api/runs":
            self._send_json({"runs": list_runs(self.runs_dir)})
        elif path == "/api/loss-history":
            run = query.get("run", [""])[0]
            if not run:
                self._send_json({"error": "run parameter required"}, 400)
                return
            self._send_json({"history": get_loss_history(self.runs_dir, run)})
        elif path.startswith("/api/eval/"):
            # /api/eval/<run>/<epoch>
            parts = path.split("/")
            if len(parts) < 5:
                self.send_error(400, "Expected /api/eval/<run>/<epoch>")
                return
            run = parts[3]
            try:
                epoch = int(parts[4])
            except ValueError:
                self.send_error(400, "Invalid epoch number")
                return
            self._send_json(get_eval_metrics(self.runs_dir, run, epoch))
        elif path.startswith("/api/gallery/"):
            # /api/gallery/<run>/<epoch>/<probe_idx>
            # /api/gallery/<run>/<epoch>/tq/<query_idx>
            parts = path.split("/")
            if len(parts) < 6:
                self.send_error(400, "Expected /api/gallery/<run>/<epoch>/...")
                return
            run = parts[3]
            try:
                epoch = int(parts[4])
            except ValueError:
                self.send_error(400, "Invalid epoch number")
                return

            # Text query gallery: /api/gallery/<run>/<epoch>/tq/<query_idx>
            if len(parts) >= 7 and parts[5] == "tq":
                try:
                    query_idx = int(parts[6])
                except ValueError:
                    self.send_error(400, "Invalid query index")
                    return
                tq_dir = self.runs_dir / run / "eval" / f"epoch_{epoch:03d}_tq"
                tq_path = tq_dir / f"query_{query_idx:03d}.jpg"
                if not tq_path.exists():
                    self.send_error(404, "Text query gallery image not found")
                    return
                self._serve_image(tq_path)
                return

            # Probe gallery: /api/gallery/<run>/<epoch>/<probe_idx>
            try:
                probe_idx = int(parts[5])
            except ValueError:
                self.send_error(400, "Invalid probe index")
                return
            gallery_dir = self.runs_dir / run / "eval" / f"epoch_{epoch:03d}_nn"
            gallery_path = gallery_dir / f"probe_{probe_idx:03d}.jpg"
            if not gallery_path.exists():
                # Fallback: older runs may use val-set indices instead of
                # sequential numbering.  List files and serve by position.
                try:
                    probe_files = sorted(
                        gallery_dir.glob("probe_*.jpg"),
                        key=lambda p: int(p.stem.split("_")[1]),
                    )
                    if probe_idx < len(probe_files):
                        gallery_path = probe_files[probe_idx]
                    else:
                        self.send_error(404, "Probe image not found")
                        return
                except (ValueError, OSError):
                    self.send_error(404, "Probe image not found")
                    return
            self._serve_image(gallery_path)
        elif path == "/api/image":
            # /api/image?path=<relative_path>
            rel_path = query.get("path", [""])[0]
            if not rel_path:
                self.send_error(400, "path parameter required")
                return
            # Security: resolve within project root only
            full_path = (PROJECT_ROOT / rel_path).resolve()
            if not str(full_path).startswith(str(PROJECT_ROOT)):
                self.send_error(403, "Path outside project root")
                return
            self._serve_image(full_path)
        elif path == "/api/csv-sample":
            raw_csv = query.get("path", [""])[0]
            if raw_csv and not Path(raw_csv).is_absolute():
                csv_path = str((PROJECT_ROOT / raw_csv).resolve())
            else:
                csv_path = raw_csv or str(DEFAULT_CSV)
            if not csv_path or not Path(csv_path).exists():
                self._send_json({"error": "CSV not found"}, 404)
                return
            try:
                with open(csv_path, encoding="utf-8") as f:
                    f.readline()  # skip header
                    first_row = f.readline().strip()
                    row_count = sum(1 for _ in f) + (1 if first_row else 0)
                if first_row.startswith('"'):
                    sample_path = first_row.split('"')[1]
                else:
                    sample_path = first_row.split(",")[0]
                image_exists = Path(sample_path).exists()
                self._send_json({
                    "csv_path": csv_path,
                    "sample_path": sample_path,
                    "row_count": row_count,
                    "image_exists": image_exists,
                })
            except Exception as e:
                self._send_json({"error": str(e)}, 500)
        elif path == "/api/path-exists":
            check_path = query.get("path", [""])[0]
            if not check_path:
                self._send_json({"error": "path parameter required"}, 400)
                return
            self._send_json({"path": check_path, "exists": Path(check_path).exists()})
        elif path == "/api/gpu":
            self._send_json(get_gpu_stats())
        elif path == "/api/processes":
            self._send_json({
                "training": {
                    "running": training_manager.is_running,
                    "pid": training_manager.pid,
                    "exit_code": training_manager.exit_code,
                },
            })
        elif path == "/api/logs/training":
            n = int(query.get("n", ["200"])[0])
            self._send_json({"lines": training_manager.get_log(n)})
        else:
            self.send_error(404)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        # Read request body
        content_len = int(self.headers.get("Content-Length", 0))
        body: dict[str, Any] = {}
        if content_len > 0:
            raw = self.rfile.read(content_len)
            try:
                body = json.loads(raw)
            except json.JSONDecodeError:
                self.send_error(400, "Invalid JSON")
                return

        if path == "/api/control":
            run = body.get("run", "")
            command = body.get("command", "")
            value = body.get("value")

            if not run:
                self._send_json({"error": "run name required"}, 400)
                return
            if command not in ("pause", "stop", "set_lr", "resume"):
                self._send_json({"error": f"Unknown command: {command}"}, 400)
                return

            control_path = self.runs_dir / run / "control.json"
            if not control_path.parent.is_dir():
                self._send_json({"error": f"Run '{run}' not found"}, 404)
                return

            # Write the command for the trainer to pick up
            control_data: dict[str, Any] = {"command": command}
            if value is not None:
                control_data["value"] = value

            try:
                control_path.write_text(
                    json.dumps(control_data), encoding="utf-8",
                )
                self._send_json({"status": "ok", "command": command})
            except OSError as e:
                self._send_json({"error": str(e)}, 500)

        elif path == "/api/train/launch":
            csv_path = body.get("csv", str(DEFAULT_CSV))
            cmd_args = ["--csv", csv_path]
            if body.get("run_name"):
                cmd_args.extend(["--run-name", body["run_name"]])
            if body.get("epochs"):
                cmd_args.extend(["--epochs", str(body["epochs"])])
            if body.get("lr"):
                cmd_args.extend(["--lr", str(body["lr"])])
            if body.get("lora_rank"):
                cmd_args.extend(["--lora-rank", str(body["lora_rank"])])
            if body.get("lora_alpha"):
                cmd_args.extend(["--lora-alpha", str(body["lora_alpha"])])
            if body.get("patience"):
                cmd_args.extend(["--patience", str(body["patience"])])
            if body.get("val_fraction"):
                cmd_args.extend(["--val-fraction", str(body["val_fraction"])])
            if body.get("resume"):
                cmd_args.extend(["--resume", body["resume"]])
            if body.get("batch_size"):
                cmd_args.extend(["--batch-size", str(body["batch_size"])])
            if body.get("eval_interval"):
                cmd_args.extend(["--eval-interval", str(body["eval_interval"])])
            if body.get("path_remap"):
                cmd_args.extend(["--path-remap", body["path_remap"]])
            if body.get("base_dir"):
                cmd_args.extend(["--base-dir", body["base_dir"]])
            elif str(self.runs_dir) != str(DEFAULT_RUNS_DIR):
                # Auto-pass dashboard's --runs-dir to trainer if non-default
                cmd_args.extend(["--base-dir", str(self.runs_dir)])
            try:
                result = training_manager.launch(cmd_args)
                self._send_json(result)
            except RuntimeError as e:
                self._send_json({"error": str(e)}, 409)

        elif path == "/api/train/stop":
            self._send_json(training_manager.stop())

        else:
            self.send_error(404)

    # ── Response helpers ──────────────────────────────────────────────

    def _serve_html(self) -> None:
        try:
            content = HTML_FILE.read_bytes()
        except FileNotFoundError:
            self.send_error(404, "Dashboard HTML not found")
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(content)

    def _send_json(self, data: Any, status: int = 200) -> None:
        body = _json_safe(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _serve_image(self, path: Path) -> None:
        if not path.exists():
            self.send_error(404)
            return
        mime = mimetypes.guess_type(str(path))[0] or "image/jpeg"
        try:
            content = path.read_bytes()
        except OSError:
            self.send_error(500)
            return
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "public, max-age=3600")
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, format: str, *args: object) -> None:
        """Suppress default HTTP access logging."""
        pass


# ── CLI ──────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA Training Dashboard")
    parser.add_argument(
        "--port", type=int, default=9874,
        help="HTTP server port (default: 9874)",
    )
    parser.add_argument(
        "--runs-dir", type=str, default=str(DEFAULT_RUNS_DIR),
        help="Directory containing training runs",
    )
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)

    DashboardHandler.runs_dir = runs_dir

    server = ThreadedServer(("0.0.0.0", args.port), DashboardHandler)
    server.runs_dir = runs_dir

    print(f"Training Dashboard: http://localhost:{args.port}")
    print(f"Watching: {runs_dir}")
    print("Press Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        print("\nShutting down — stopping child processes…")
        if training_manager.is_running:
            training_manager.stop()
            print("  Training process stopped.")
        print("Done.")


if __name__ == "__main__":
    main()
