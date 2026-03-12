"""Lightweight HTTP webhook server for push notifications from the scheduler."""

import json
import queue
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import List


class _NotifyHandler(BaseHTTPRequestHandler):
    """Handles POST /notify requests and puts messages into the server's queue."""

    def do_POST(self) -> None:
        if self.path != "/notify":
            self.send_response(404)
            self.end_headers()
            return

        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        try:
            data = json.loads(body)
        except (json.JSONDecodeError, ValueError):
            self.send_response(400)
            self.end_headers()
            return

        task_id = data.get("id", "")
        description = data.get("description", "")
        delay_seconds = data.get("delay_seconds", 0)

        if delay_seconds and delay_seconds >= 60:
            elapsed = f"{int(delay_seconds) // 60} мин"
        else:
            elapsed = f"{int(delay_seconds)} сек"

        message = f"[REMINDER] прошло {elapsed}, Описание: {description} (id={task_id})"
        self.server.notification_queue.put(message)  # type: ignore[attr-defined]

        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{"status":"ok"}')

    def log_message(self, *args) -> None:  # type: ignore[override]
        pass  # suppress access logs


class NotificationServer:
    """Runs a daemon HTTP server that collects webhook notifications."""

    def __init__(self, port: int = 8088) -> None:
        self._port = port
        self._httpd: HTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._queue: queue.Queue = queue.Queue()

    def start(self) -> None:
        """Start the HTTP server in a background daemon thread."""
        httpd = HTTPServer(("localhost", self._port), _NotifyHandler)
        httpd.notification_queue = self._queue  # type: ignore[attr-defined]
        self._httpd = httpd

        self._thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Shut down the HTTP server."""
        if self._httpd is not None:
            self._httpd.shutdown()

    def get_url(self) -> str:
        return f"http://localhost:{self._port}/notify"

    def check_notifications(self) -> List[str]:
        """Drain the notification queue and return all pending messages."""
        messages: List[str] = []
        while True:
            try:
                messages.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return messages
