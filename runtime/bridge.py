"""Loopback bridge between POM's plugin proxy and the ComfyUI server.

ComfyUI has no authentication of its own, so it listens on a random loopback
port that only this bridge knows, and the bridge refuses every request that
does not carry the per-launch `x-pom-plugin-token` POM attaches. The bridge
also ends ComfyUI's whole process group when the plugin closes.

    python -I bridge.py <base> <source> <home>
"""
import http.client
import http.server
import json
import os
from pathlib import Path
import secrets
import select
import signal
import socket
import subprocess
import sys
import threading
import time

PREFIX = "/api/ui/plugins/comfyui/proxy"
MAX_BODY = 2 * 1024 * 1024 * 1024
HOP_BY_HOP = ("host", "connection", "keep-alive", "x-pom-plugin-token", "authorization", "cookie",
              "x-forwarded-prefix", "content-length", "transfer-encoding", "upgrade", "origin")


class Bridge(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_args):
        pass

    def handle_request(self):
        token = self.headers.get("x-pom-plugin-token", "")
        if not secrets.compare_digest(token, self.server.secret):
            self.send_error(401)
            return
        target = self.server.target
        # POM authenticates the browser and strips its credentials; ComfyUI
        # refuses a loopback request whose Origin does not match its Host, so
        # both name the loopback server.
        headers = {key: value for key, value in self.headers.items() if key.lower() not in HOP_BY_HOP}
        headers["Host"] = f"127.0.0.1:{target}"
        if self.headers.get("Origin"):
            headers["Origin"] = f"http://127.0.0.1:{target}"
        headers["X-Forwarded-Prefix"] = PREFIX
        if self.headers.get("Upgrade", "").lower() == "websocket":
            self.websocket(headers, target)
            return
        chunked = self.headers.get("Transfer-Encoding", "").lower() == "chunked"
        if self.headers.get("Transfer-Encoding") and not chunked:
            self.send_error(400)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self.send_error(400)
            return
        if length < 0 or length > MAX_BODY or (chunked and self.headers.get("Content-Length")):
            self.send_error(413 if length > MAX_BODY else 400)
            return
        if chunked:
            body = self.chunked_body()
        elif length:
            body = self.sized_body(length)
            headers["Content-Length"] = str(length)
        else:
            body = None
        connection = http.client.HTTPConnection("127.0.0.1", target, timeout=600)
        try:
            connection.request(self.command, self.path, body, headers, encode_chunked=chunked)
            response = connection.getresponse()
            self.send_response_only(response.status, response.reason)
            for key, value in response.getheaders():
                if key.lower() not in ("transfer-encoding", "connection", "content-length", "set-cookie",
                                       "x-frame-options"):
                    self.send_header(key, value)
            self.send_header("Transfer-Encoding", "chunked")
            self.send_header("Connection", "close")
            self.end_headers()
            while block := response.read1(64 * 1024):
                self.wfile.write(f"{len(block):x}\r\n".encode() + block + b"\r\n")
                self.wfile.flush()
            self.wfile.write(b"0\r\n\r\n")
        except (OSError, ConnectionError, ValueError):
            pass
        finally:
            connection.close()
            self.close_connection = True

    def sized_body(self, length):
        total = 0
        while total < length:
            block = self.rfile.read(min(length - total, 64 * 1024))
            if not block:
                raise ValueError("truncated upload")
            total += len(block)
            yield block

    def chunked_body(self):
        # POM streams request bodies chunked; pass them on without buffering
        # a whole upload (images, models) in memory.
        total = 0
        while True:
            line = self.rfile.readline(100)
            size = int(line.split(b";", 1)[0].strip(), 16)
            if size < 0 or total + size > MAX_BODY:
                raise ValueError("upload too large")
            if size == 0:
                while self.rfile.readline(8192) not in (b"\r\n", b"\n", b""):
                    pass
                return
            remaining = size
            while remaining:
                block = self.rfile.read(min(remaining, 64 * 1024))
                if not block:
                    raise ValueError("truncated upload")
                remaining -= len(block)
                total += len(block)
                yield block
            if self.rfile.read(2) != b"\r\n":
                raise ValueError("invalid chunk framing")

    def websocket(self, headers, target):
        # POM forwards lowercase header names; a second Upgrade header would
        # make aiohttp refuse the handshake, so each is sent once.
        upstream = socket.create_connection(("127.0.0.1", target), timeout=10)
        try:
            headers["Upgrade"] = "websocket"
            headers["Connection"] = "Upgrade"
            request = f"{self.command} {self.path} HTTP/1.1\r\n" + "".join(
                f"{k}: {v}\r\n" for k, v in headers.items()) + "\r\n"
            upstream.sendall(request.encode())
            response = b""
            while b"\r\n\r\n" not in response and len(response) < 65536:
                block = upstream.recv(4096)
                if not block:
                    return
                response += block
            self.connection.sendall(response)
            if not response.startswith(b"HTTP/1.1 101"):
                return
            self.close_connection = True
            upstream.settimeout(None)
            for sock in (self.connection, upstream):
                sock.setblocking(False)
            while True:
                ready, _, _ = select.select((self.connection, upstream), (), (), 30)
                for sock in ready:
                    try:
                        block = sock.recv(65536)
                        if not block:
                            return
                        (upstream if sock is self.connection else self.connection).sendall(block)
                    except BlockingIOError:
                        continue
                    except (OSError, ConnectionError):
                        return
        finally:
            upstream.close()

    do_GET = do_POST = do_PUT = do_PATCH = do_DELETE = do_OPTIONS = do_HEAD = handle_request


def stop_comfy(child, grace=5.0):
    """End ComfyUI and anything it spawned (its own session/process group)."""
    if child.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(child.pid)], capture_output=True)
        child.wait()
        return
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        child.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.wait()


def watch(child):
    """The supervisor watches the bridge: a ComfyUI that died must end it."""
    while child.poll() is None:
        time.sleep(2)
    if os.name == "nt":
        os._exit(1)
    os.kill(os.getpid(), signal.SIGTERM)


def comfy_command(source, home, port):
    extra = [arg for arg in os.environ.get("POM_COMFYUI_ARGS", "").split("\n") if arg]
    return [sys.executable, str(source / "main.py"), "--listen", "127.0.0.1", "--port", str(port),
            "--base-directory", str(home), "--disable-auto-launch", *extra]


def serve(base, source, home):
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    env = {key: value for key, value in os.environ.items() if key != "POM_COMFYUI_ARGS"}
    env["PYTHONUNBUFFERED"] = "1"
    log = (base / "comfyui.log").open("ab", buffering=0)
    child = subprocess.Popen(comfy_command(source, home, port), cwd=source, env=env,
                             stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                             start_new_session=(os.name != "nt"),
                             creationflags=(subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0))
    try:
        # The first start of a new install compiles and imports a lot.
        for _ in range(600):
            if child.poll() is not None:
                raise RuntimeError("ComfyUI exited; see comfyui.log")
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                    break
            except OSError:
                time.sleep(1)
        else:
            raise RuntimeError("ComfyUI did not start; see comfyui.log")
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Bridge)
        server.daemon_threads = True
        server.secret = secrets.token_hex(32)
        server.target = port
        print(json.dumps({"status": "ready", "port": server.server_port, "token": server.secret}), flush=True)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        threading.Thread(target=watch, args=(child,), daemon=True).start()
        # The supervisor closes stdin to stop us.
        while sys.stdin.buffer.read(1):
            pass
        server.shutdown()
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        stop_comfy(child)
        log.close()


if __name__ == "__main__":
    serve(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
