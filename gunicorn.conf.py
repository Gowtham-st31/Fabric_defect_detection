# Gunicorn configuration — auto-discovered when gunicorn runs from this directory.
# Render start command: gunicorn app:app --bind 0.0.0.0:$PORT
# (workers, threads, timeout can be overridden via CLI flags or env vars)

import multiprocessing
import os

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
