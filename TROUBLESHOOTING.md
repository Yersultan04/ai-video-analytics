# Telegram bot troubleshooting

## 1) `telegram.error.NetworkError: httpx.ReadError`

### What it means
This is usually a transient network read interruption while long-polling `getUpdates`.
It does **not** necessarily mean your bot token is invalid or your app is broken.

### What to do
- Start the bot again (or let your process manager restart it).
- Verify the network path to `api.telegram.org` is stable.
- If possible, use a retry/backoff policy and keep polling enabled.

---

## 2) `RuntimeError: There is no current event loop in thread 'asyncio_0'`

### Root cause
Your video analysis runs in a worker thread via `run_in_executor(...)`.
Inside that worker thread, the progress callback called `asyncio.get_event_loop()`.
In modern Python, worker threads do not automatically have a current event loop,
which raises this runtime error.

### Correct fix
Capture the main event loop in the async handler and schedule progress updates
thread-safely from the worker thread:

- Capture loop with `loop = asyncio.get_running_loop()`.
- In the callback, use `asyncio.run_coroutine_threadsafe(...)` with that loop.
- Keep progress edits best-effort so temporary Telegram API edit failures do not crash analysis.

This pattern is now implemented in `src/bot.py`.

---

## 3) `ultralytics` auto-installs `lap` and asks for restart

### What it means
`ultralytics` installed `lap` at runtime and warns that restart/rerun may be needed.

### Recommended setup
Install dependencies before starting the bot:

```bash
python -m pip install -U pip
python -m pip install "lap>=0.5.12"
```

Optionally pin in your `requirements.txt`:

```txt
lap>=0.5.12
```

---

## 4) `git add .` staged huge local files (`.env`, videos, model weights)

### What to do now
If you already staged everything locally, unstage first and only stage source files:

```bash
git reset
git add src/bot.py src/analyze.py TROUBLESHOOTING.md .gitignore
```

### Prevent it next time
Use `.gitignore` (already added in this repo) for local artifacts:
- `.env`
- `*.mp4`, `*.part`
- `*.pt`
- `__pycache__/`
