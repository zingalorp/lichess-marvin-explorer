"""Chrome/Edge native messaging with a bounded, latest-position-per-tab queue."""
import json
import os
import struct
import sys
import threading
from collections import OrderedDict

from explorer import Cancelled, Explorer

MAX_MESSAGE = 256 * 1024


def read_exact(stream, size):
    data = bytearray()
    while len(data) < size:
        part = stream.read(size-len(data))
        if not part:
            if not data:
                return None
            raise EOFError("Truncated native message")
        data.extend(part)
    return bytes(data)


def read_message(stream):
    header = read_exact(stream, 4)
    if header is None:
        return None
    size = struct.unpack("<I", header)[0]
    if not 0 < size <= MAX_MESSAGE:
        raise ValueError("Invalid native message size")
    payload = read_exact(stream, size)
    if payload is None:
        raise EOFError("Missing native message body")
    msg = json.loads(payload)
    if not isinstance(msg, dict):
        raise ValueError("Native message must be an object")
    return msg


def write_message(stream, message):
    payload = json.dumps(message, separators=(",", ":"), allow_nan=False).encode()
    if len(payload) > 1024 * 1024:
        raise ValueError("Native response is too large")
    stream.write(struct.pack("<I", len(payload))+payload)
    stream.flush()


def main():
    if os.name == "nt":
        import msvcrt
        msvcrt.setmode(sys.stdin.fileno(), os.O_BINARY)
        msvcrt.setmode(sys.stdout.fileno(), os.O_BINARY)
    output = sys.stdout.buffer
    # Third-party diagnostic prints must never corrupt the native protocol.
    sys.stdout = sys.stderr
    # On Windows, loading NumPy's native extension while another thread blocks
    # on the stdin pipe can deadlock. Import DLL-backed dependencies BEFORE the
    # reader starts; keep the expensive ONNX session itself lazy.
    startup_error = None
    try:
        import numpy  # noqa: F401
        import onnxruntime  # noqa: F401
        import marvin.inference  # noqa: F401
    except Exception as exc:
        startup_error = f'Marvin dependencies could not load: {exc}. Install host/requirements.txt.'
    explorer = Explorer()
    pending = OrderedDict()
    condition = threading.Condition()
    write_lock = threading.Lock()
    closed = False
    # The request being evaluated. Any newer message from its tab (a new position
    # or a cancel) stops it between WDL batches.
    current = {"client": None, "cancelled": False}

    def send(message):
        with write_lock:
            write_message(output, message)

    def reader():
        nonlocal closed
        try:
            while (message := read_message(sys.stdin.buffer)) is not None:
                if message.get("type") == "info":
                    send({"id": message.get("id"), "ok": True, "result": explorer.info()})
                    continue
                client = str(message.get("client", "default"))[:100]
                with condition:
                    if current["client"] == client:
                        current["cancelled"] = True
                    previous = pending.pop(client, None)
                    if previous:
                        send({"id": previous.get("id"), "ok": False, "error": "Superseded by a newer position."})
                    if message.get("type") == "cancel":
                        continue
                    if len(pending) >= 32:
                        send({"id": message.get("id"), "ok": False, "error": "Host is busy. Try again."})
                        continue
                    pending[client] = message
                    condition.notify()
        except (ValueError, EOFError, OSError) as exc:
            print(str(exc), file=sys.stderr)
        finally:
            with condition:
                closed = True
                condition.notify()

    threading.Thread(target=reader, daemon=True).start()
    while True:
        with condition:
            condition.wait_for(lambda: pending or closed)
            if closed:
                return
            client, message = pending.popitem(last=False)
            job = {"client": client, "cancelled": False}
            current = job
        try:
            if message.get("type") != "analyse":
                raise ValueError("Unknown native request.")
            if startup_error:
                raise ValueError(startup_error)
            request_id = message.get("id")
            result = explorer.analyse(
                message.get("position"), message.get("filters"),
                # Policy rows first, then WDL as each batch completes.
                progress=lambda partial: send({"id": request_id, "ok": True, "partial": True, "result": partial}),
                cancelled=lambda: job["cancelled"])
            send({"id": request_id, "ok": True, "result": result})
        except Cancelled as exc:
            send({"id": message.get("id"), "ok": False, "error": str(exc)})
        except Exception as exc:
            print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
            send({"id": message.get("id"), "ok": False, "error": str(exc)[:600]})


if __name__ == "__main__":
    main()
