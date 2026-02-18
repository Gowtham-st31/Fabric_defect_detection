# Gunicorn configuration — auto-discovered when gunicorn runs from this directory.
#
# Render start command (set in Dashboard → Settings → Start Command):
#   gunicorn app:app -c gunicorn.conf.py --bind 0.0.0.0:$PORT
#
# IMPORTANT: Do NOT add --workers / --threads / --timeout on the command line;
# they override this file.  Use env vars instead if needed.

import os

# --- Pre-load the app BEFORE forking workers -----------------------------
# This runs _preload_models() (+ warmup) once in the master process, so
# workers are ready to serve immediately.  On a single-worker setup the
# effect is the same: models are loaded before the port is opened.
preload_app = True

# --- Workers / threads ---------------------------------------------------
# Render free tier has 512 MB RAM; keep 1 worker + 2 threads to avoid OOM
# while allowing a health-check to succeed during a slow inference.
workers = int(os.environ.get("WEB_CONCURRENCY", "1"))
threads = int(os.environ.get("GUNICORN_THREADS", "2"))

# --- Timeouts ------------------------------------------------------------
# Model loading can take 30-40 s on cold start; give the worker plenty of
# time to boot and handle slow inferences.
timeout = int(os.environ.get("GUNICORN_TIMEOUT", "180"))
graceful_timeout = 30

# --- Keep-alive -----------------------------------------------------------
keepalive = 5

# --- Logging --------------------------------------------------------------
accesslog = "-"
errorlog = "-"
loglevel = os.environ.get("GUNICORN_LOG_LEVEL", "info")
