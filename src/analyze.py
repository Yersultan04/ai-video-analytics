"""Video analysis helpers."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable


def analyze_video(video_path: Path, progress_callback: Callable[[int], None] | None = None) -> str:
    """Analyze a video and report progress.

    This lightweight implementation exists so the bot module can be executed and
    tested in isolation. Replace with real model inference as needed.
    """
    for pct in (10, 35, 60, 85, 100):
        time.sleep(0.01)
        if progress_callback is not None:
            progress_callback(pct)

    return f"Processed: {video_path.name}"
