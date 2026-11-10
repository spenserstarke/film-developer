"""
Finishing window: serves a local page (127.0.0.1 only) with a real-time creative-look preview.

  python3 finish_server.py photo1_kodak_portra_400.tif photo2_....tif

Prints "FINISH_URL http://127.0.0.1:<port>/" once ready. The page previews looks on the GPU (WebGL);
Save applies the identical maths (looks.apply_look) to the full-resolution 16-bit TIFF and writes
<name>_<look>.tif next to it, leaving the un-graded film TIFF untouched. The server exits when you
click Done, or a couple of minutes after the page is closed.
"""
import io
import json
import os
import sys
import threading
import time
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

import numpy as np
import tifffile
from PIL import Image

import looks

HERE = os.path.dirname(os.path.abspath(__file__))
PREVIEW_LONG = 1600
STATE = {"last_ping": None, "started": time.time(), "saving": False}


def srgb_icc():
    from PIL import ImageCms
    return ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()


class Photos:
    def __init__(self, paths):
        self.paths = [os.path.abspath(p) for p in paths]
        self._prev = {}

    def full(self, i):
        a = tifffile.imread(self.paths[i])
        return a.astype(np.float32) / (65535.0 if a.dtype == np.uint16 else 255.0)

    def preview_png(self, i):
        if i not in self._prev:
            a = self.full(i)
            im = Image.fromarray((np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8))
            s = PREVIEW_LONG / max(im.size)
            if s < 1:
                im = im.resize((round(im.size[0] * s), round(im.size[1] * s)), Image.LANCZOS)
            buf = io.BytesIO()
            im.save(buf, "PNG")
            self._prev[i] = buf.getvalue()
        return self._prev[i]

    def info(self):
        return [{"index": i, "name": os.path.basename(p)} for i, p in enumerate(self.paths)]

    def save(self, i, params):
        lib = looks.load_presets()
        preset = next(p for p in lib["presets"] if p["id"] == params.get("preset", "none"))
        out = looks.apply_look(self.full(i), preset, float(params.get("strength", 1.0)),
                               float(params.get("warmth", 0)), float(params.get("tint", 0)),
                               float(params.get("saturation", 0)), float(params.get("contrast", 0)),
                               float(params.get("fade", 0)), lib=lib)
        stem = os.path.splitext(self.paths[i])[0]
        path = f"{stem}_{preset['id']}.tif"
        tifffile.imwrite(path, (out * 65535 + 0.5).astype(np.uint16), photometric="rgb",
                         compression="zlib", iccprofile=srgb_icc())
        return path


def make_handler(photos):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code=200):
            self._send(code, json.dumps(obj).encode(), "application/json")

        def do_GET(self):
            STATE["last_ping"] = time.time()
            if self.path in ("/", "/index.html"):
                self._send(200, open(os.path.join(HERE, "finish_ui.html"), "rb").read(), "text/html; charset=utf-8")
            elif self.path == "/api/photos":
                self._json(photos.info())
            elif self.path == "/api/looks":
                self._json(looks.load_presets())
            elif self.path.startswith("/preview/"):
                i = int(self.path.split("/")[2].split(".")[0])
                self._send(200, photos.preview_png(i), "image/png")
            elif self.path == "/api/ping":
                self._json({"ok": True})
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):
            STATE["last_ping"] = time.time()
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}")
            if self.path == "/api/save":
                try:
                    STATE["saving"] = True
                    path = photos.save(int(body["index"]), body)
                    print(f"SAVED {path}", flush=True)
                    self._json({"saved": os.path.basename(path), "path": path})
                except Exception as e:                       # report to the page, keep running
                    self._json({"error": str(e)}, 500)
                finally:
                    STATE["saving"] = False
            elif self.path == "/api/done":
                self._json({"bye": True})
                threading.Thread(target=lambda: (time.sleep(0.3), os._exit(0)), daemon=True).start()
            else:
                self._send(404, b"not found", "text/plain")
    return H


def main():
    paths = [p for p in sys.argv[1:] if os.path.exists(p)]
    if not paths:
        print("no photos", flush=True)
        return 1
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(Photos(paths)))
    print(f"FINISH_URL http://127.0.0.1:{srv.server_address[1]}/", flush=True)

    def watchdog():                      # page closed without Done -> exit after 3 minutes of silence
        while True:
            time.sleep(10)
            last = STATE["last_ping"] or STATE["started"]
            if not STATE["saving"] and time.time() - last > 180:
                print("FINISH_TIMEOUT", flush=True)
                os._exit(0)
    threading.Thread(target=watchdog, daemon=True).start()
    srv.serve_forever()


if __name__ == "__main__":
    sys.exit(main() or 0)
