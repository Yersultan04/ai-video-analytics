# Telegram bot troubleshooting

## 1) `telegram.error.NetworkError: httpx.ReadError`

### What it means
Usually a transient network interruption while long-polling `getUpdates`.

### What to do
- Restart bot process.
- Check connectivity to `api.telegram.org`.
- Keep bot under a process manager for auto-restart.

---

## 2) `RuntimeError: There is no current event loop in thread 'asyncio_0'`

### Root cause
Analysis runs in `run_in_executor(...)`; worker thread callback tried to touch asyncio loop directly.

### Fix implemented
- Main loop captured with `asyncio.get_running_loop()`.
- Worker thread schedules coroutine updates with `asyncio.run_coroutine_threadsafe(...)`.

---

## 3) Missing runtime dependencies for MVP AI report

This MVP requires:
- `python-telegram-bot`
- `ultralytics`
- `opencv-python`
- `matplotlib`

Install:

```bash
python -m pip install -U pip
python -m pip install python-telegram-bot ultralytics opencv-python matplotlib
```

If `ultralytics` asks for `lap`:

```bash
python -m pip install "lap>=0.5.12"
```

---

## 4) Accidentally staged huge local files (`git add .`)

```bash
git reset
git restore .
git clean -fd
```

Then stage only project files:

```bash
git add src/bot.py src/analyze.py TROUBLESHOOTING.md .gitignore
```
