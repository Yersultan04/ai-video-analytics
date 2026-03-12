"""Video analysis helpers."""

from __future__ import annotations

from importlib import import_module
from importlib.util import find_spec
from pathlib import Path
from typing import Callable


def _emit_progress(progress_callback: Callable[[int], None] | None, pct: int) -> None:
    if progress_callback is not None:
        progress_callback(pct)


def analyze_video(video_path: Path, progress_callback: Callable[[int], None] | None = None) -> str:
    """Analyze a video file and report lightweight metadata.

    The function stays dependency-light: it always returns basic file info and,
    when OpenCV is available in the environment, it adds resolution/fps/duration.
    """
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video file was not found: {path}")

    _emit_progress(progress_callback, 10)

    file_size_mb = path.stat().st_size / (1024 * 1024)
    suffix = path.suffix.lower() or "(no extension)"

    _emit_progress(progress_callback, 35)

    width = None
    height = None
    fps = None
    duration_s = None

    cv2_spec = find_spec("cv2")
    if cv2_spec is not None:
        cv2 = import_module("cv2")
        cap = cv2.VideoCapture(str(path))
        try:
            if cap.isOpened():
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
                fps_value = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
                frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
                if fps_value > 0:
                    fps = fps_value
                    duration_s = frame_count / fps_value
        finally:
            cap.release()

    _emit_progress(progress_callback, 70)

    lines = [
        f"Processed: {path.name}",
        f"Format: {suffix}",
        f"Size: {file_size_mb:.2f} MB",
    ]

    if width and height:
        lines.append(f"Resolution: {width}x{height}")
    if fps is not None:
        lines.append(f"FPS: {fps:.2f}")
    if duration_s is not None:
        lines.append(f"Duration: {duration_s:.2f} sec")

    if cv2_spec is None:
        lines.append("Note: Install opencv-python for detailed metadata (resolution/fps/duration).")

    _emit_progress(progress_callback, 100)

    return "\n".join(lines)
