import os

bind = "0.0.0.0:8000"
# One model per process. Do not increase workers on a single GPU.
workers = 1
worker_class = "gthread"
threads = 4
timeout = int(os.getenv("GUNICORN_TIMEOUT", "600"))
graceful_timeout = 60
keepalive = 5
preload_app = False
accesslog = "-"
errorlog = "-"
