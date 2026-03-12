"""Telegram bot entrypoint.

This module includes a thread-safe progress callback bridge used while video
analysis is executed in a thread pool.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Callable

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from .analyze import analyze_video

logger = logging.getLogger(__name__)


async def _edit_progress_message(status_message, pct: int) -> None:
    """Update a Telegram status message with current analysis progress."""
    try:
        await status_message.edit_text(f"Analyzing video… {pct}%")
    except Exception:  # best effort progress update
        logger.exception("Failed to update progress message")


def _build_threadsafe_progress_callback(
    *,
    loop: asyncio.AbstractEventLoop,
    status_message,
) -> Callable[[int], None]:
    """Return a progress callback safe to call from worker threads.

    `analyze_video` is executed with `run_in_executor`, so its progress callback
    runs in a non-async worker thread. Calling `asyncio.get_event_loop()` there
    fails on modern Python versions because no loop is attached to that thread.

    We solve this by capturing the running main loop and scheduling coroutine
    execution with `asyncio.run_coroutine_threadsafe(...)`.
    """

    def on_progress(pct: int) -> None:
        asyncio.run_coroutine_threadsafe(
            _edit_progress_message(status_message, pct),
            loop,
        )

    return on_progress


async def handle_video(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle incoming video/document uploads and run analysis."""
    if not update.message:
        return

    video = update.message.video
    document = update.message.document

    # Telegram documents can be arbitrary files; only accept video documents.
    if video is None and (document is None or not (document.mime_type or "").startswith("video/")):
        await update.message.reply_text("Please send a video file.")
        return

    telegram_file = video or document
    status_message = await update.message.reply_text("Downloading video…")

    file_obj = await telegram_file.get_file()
    with NamedTemporaryFile(prefix="tg_video_", suffix=".mp4", delete=False) as tmp_file:
        tmp_path = Path(tmp_file.name)
    await file_obj.download_to_drive(str(tmp_path))

    await status_message.edit_text("Analyzing video… 0%")

    loop = asyncio.get_running_loop()
    progress_callback = _build_threadsafe_progress_callback(
        loop=loop,
        status_message=status_message,
    )

    try:
        result = await loop.run_in_executor(
            None,
            lambda: analyze_video(tmp_path, progress_callback=progress_callback),
        )
        await status_message.edit_text(f"✅ Analysis done\n\n{result}")
    except Exception:
        logger.exception("Analysis failed")
        await status_message.edit_text("❌ Analysis failed. Please try again.")
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message:
        await update.message.reply_text("Send me a video and I will analyze it.")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s — %(name)s — %(levelname)s — %(message)s",
    )

    import os

    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set")

    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.VIDEO | filters.Document.VIDEO, handle_video))

    logger.info("CafeEye bot started")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
