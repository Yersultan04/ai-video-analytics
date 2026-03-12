"""Video analysis helpers for CafeEye MVP reporting."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Callable


@dataclass(slots=True)
class AnalysisResult:
    """Structured analysis output consumed by the Telegram bot."""

    summary_text: str
    json_path: Path
    chart_path: Path


def _emit_progress(progress_callback: Callable[[int], None] | None, pct: int) -> None:
    if progress_callback is not None:
        progress_callback(max(0, min(100, int(pct))))


def _euclidean(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _boxes_overlap(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> bool:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    return not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)


def _build_episodes(
    timeline: list[dict[str, float | int | bool]],
    flag_key: str,
) -> list[dict[str, float]]:
    episodes: list[dict[str, float]] = []
    start: float | None = None
    prev_second: float | None = None

    for row in timeline:
        second = float(row["second"])
        flag = bool(row[flag_key])
        if flag and start is None:
            start = second
        if not flag and start is not None:
            end_second = prev_second if prev_second is not None else second
            episodes.append(
                {
                    "start_sec": round(start, 2),
                    "end_sec": round(end_second, 2),
                    "duration_sec": round(max(0.0, end_second - start), 2),
                }
            )
            start = None
        prev_second = second

    if start is not None and prev_second is not None:
        episodes.append(
            {
                "start_sec": round(start, 2),
                "end_sec": round(prev_second, 2),
                "duration_sec": round(max(0.0, prev_second - start), 2),
            }
        )

    return episodes


def analyze_video(
    video_path: Path,
    progress_callback: Callable[[int], None] | None = None,
    output_dir: Path | None = None,
) -> AnalysisResult:
    """Run a practical MVP analysis: people/phone/idle timeline + chart + JSON."""
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video file was not found: {path}")

    cv2 = import_module("cv2")
    plt = import_module("matplotlib.pyplot")
    YOLO = import_module("ultralytics").YOLO

    _emit_progress(progress_callback, 5)

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError("Could not open the uploaded video file.")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if fps <= 0:
        fps = 25.0

    step = max(1, int(round(fps)))  # analyze roughly 1 frame per second
    sample_seconds = step / fps
    total_steps = max(1, total_frames // step)
    video_duration_sec = total_frames / fps if total_frames > 0 else 0.0

    # Workplace ROI: lower-center area where cashier typically stands.
    roi_x1 = int(width * 0.35)
    roi_x2 = int(width * 0.75)
    roi_y1 = int(height * 0.45)
    roi_y2 = int(height * 0.98)

    # Line crossing for entrance/exit estimate.
    entry_line_x = int(width * 0.5)

    model = YOLO("yolov8n.pt")

    _emit_progress(progress_callback, 15)

    tracks: dict[int, tuple[float, float]] = {}
    prev_side_by_track: dict[int, int] = {}
    seen_ids: set[int] = set()
    next_track_id = 1
    move_threshold_px = 12.0

    idle_seconds = 0.0
    phone_seconds = 0.0
    workplace_presence_seconds = 0.0
    in_count = 0
    out_count = 0
    timeline: list[dict[str, float | int | bool]] = []

    processed = 0
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % step != 0:
            frame_idx += 1
            continue

        results = model(frame, verbose=False)
        boxes = results[0].boxes

        people_boxes: list[tuple[float, float, float, float]] = []
        phone_boxes: list[tuple[float, float, float, float]] = []

        if boxes is not None:
            xyxy = boxes.xyxy.cpu().numpy().tolist()
            cls = boxes.cls.cpu().numpy().tolist()
            for box, cls_id in zip(xyxy, cls):
                x1, y1, x2, y2 = map(float, box)
                if int(cls_id) == 0:  # person
                    people_boxes.append((x1, y1, x2, y2))
                elif int(cls_id) == 67:  # cell phone
                    phone_boxes.append((x1, y1, x2, y2))

        centers = [((x1 + x2) / 2, (y1 + y2) / 2) for x1, y1, x2, y2 in people_boxes]

        new_tracks: dict[int, tuple[float, float]] = {}
        movement: list[float] = []

        for center in centers:
            best_id = None
            best_dist = float("inf")
            for track_id, prev in tracks.items():
                d = _euclidean(center, prev)
                if d < best_dist and d < 80:
                    best_dist = d
                    best_id = track_id

            if best_id is None:
                best_id = next_track_id
                next_track_id += 1
                best_dist = 0.0

            new_tracks[best_id] = center
            seen_ids.add(best_id)
            movement.append(best_dist)

            current_side = -1 if center[0] < entry_line_x else 1
            prev_side = prev_side_by_track.get(best_id)
            if prev_side is not None and prev_side != current_side:
                if prev_side == -1 and current_side == 1:
                    in_count += 1
                elif prev_side == 1 and current_side == -1:
                    out_count += 1
            prev_side_by_track[best_id] = current_side

        tracks = new_tracks

        idle_flag = False
        if movement:
            avg_move = sum(movement) / len(movement)
            idle_flag = avg_move < move_threshold_px
            if idle_flag:
                idle_seconds += sample_seconds

        phone_flag = False
        for p in phone_boxes:
            if any(_boxes_overlap(p, person) for person in people_boxes):
                phone_flag = True
                break
        if phone_flag:
            phone_seconds += sample_seconds

        workplace_flag = False
        for cx, cy in centers:
            if roi_x1 <= cx <= roi_x2 and roi_y1 <= cy <= roi_y2:
                workplace_flag = True
                break
        if workplace_flag:
            workplace_presence_seconds += sample_seconds

        second = frame_idx / fps
        timeline.append(
            {
                "second": round(second, 2),
                "people": len(people_boxes),
                "idle": idle_flag,
                "phone": phone_flag,
                "workplace": workplace_flag,
            }
        )

        processed += 1
        _emit_progress(progress_callback, 15 + (processed / total_steps) * 75)
        frame_idx += 1

    cap.release()

    if not timeline:
        raise RuntimeError("Could not read frames from video.")

    peak_row = max(timeline, key=lambda row: int(row["people"]))
    peak_second = float(peak_row["second"])

    phone_events = _build_episodes(timeline, "phone")
    idle_events = _build_episodes(timeline, "idle")

    if output_dir is None:
        output_dir = path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = path.stem
    chart_path = output_dir / f"{stem}_activity.png"
    json_path = output_dir / f"{stem}_report.json"

    x = [float(row["second"]) for row in timeline]
    y_people = [int(row["people"]) for row in timeline]
    y_idle = [1 if bool(row["idle"]) else 0 for row in timeline]
    y_phone = [1 if bool(row["phone"]) else 0 for row in timeline]
    y_workplace = [1 if bool(row["workplace"]) else 0 for row in timeline]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, y_people, label="People count", linewidth=2)
    ax.fill_between(x, y_idle, alpha=0.15, label="Idle flag")
    ax.fill_between(x, y_phone, alpha=0.15, label="Phone flag")
    ax.fill_between(x, y_workplace, alpha=0.15, label="Workplace flag")
    ax.set_title("CafeEye MVP timeline")
    ax.set_xlabel("Second")
    ax.set_ylabel("Count / Flag")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(chart_path)
    plt.close(fig)

    presence_percent = (workplace_presence_seconds / video_duration_sec * 100) if video_duration_sec > 0 else 0.0

    report = {
        "video": path.name,
        "fps": round(fps, 3),
        "total_frames": total_frames,
        "video_duration_sec": round(video_duration_sec, 2),
        "people_seen_estimate": len(seen_ids),
        "entry_count": in_count,
        "exit_count": out_count,
        "idle_seconds": round(idle_seconds, 2),
        "phone_seconds": round(phone_seconds, 2),
        "workplace_presence_seconds": round(workplace_presence_seconds, 2),
        "workplace_presence_percent": round(presence_percent, 2),
        "peak_second": round(peak_second, 2),
        "phone_events": phone_events,
        "idle_events": idle_events,
        "workplace_roi": {
            "x1": roi_x1,
            "y1": roi_y1,
            "x2": roi_x2,
            "y2": roi_y2,
        },
        "entry_line_x": entry_line_x,
        "timeline": timeline,
    }
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    phone_timecodes = "нет"
    if phone_events:
        phone_timecodes = ", ".join(
            f"{event['start_sec']:.0f}-{event['end_sec']:.0f}с" for event in phone_events[:5]
        )

    summary = (
        "📊 AI Анализ видео\n"
        f"👥 Людей замечено (оценка): {report['people_seen_estimate']}\n"
        f"🚪 Вход/выход (оценка): {report['entry_count']}/{report['exit_count']}\n"
        f"🧍 Простой: {report['idle_seconds']:.2f} сек\n"
        f"📱 Телефон: {report['phone_seconds']:.2f} сек\n"
        f"📍 У кассы: {report['workplace_presence_percent']:.1f}% времени\n"
        f"🕒 Телефон-таймкоды: {phone_timecodes}\n"
        f"⏰ Пик: {report['peak_second']:.2f} сек"
    )

    _emit_progress(progress_callback, 100)

    return AnalysisResult(summary_text=summary, json_path=json_path, chart_path=chart_path)
