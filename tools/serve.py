"""Static server for the demo and the batch harness.

Two roots, because the corpus must not live inside the repo: CFD's licence
forbids redistributing images and this repo is public.

    /          -> public/            (the pages, the vendored MediaPipe runtime)
    /corpus/   -> C:\\lab-corpus      (webcam-framed test frames, gitignored)

Also fixes the .wasm MIME type, which Python's default table gets wrong and
which MediaPipe's streaming compiler refuses without.

    .venv/Scripts/python.exe tools/serve.py [--port 8900]
"""

from __future__ import annotations

import argparse
import functools
import mimetypes
import os
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).parent.parent
PUBLIC = HERE / "public"
CORPUS = Path(r"C:\lab-corpus")

mimetypes.add_type("application/wasm", ".wasm")
mimetypes.add_type("application/javascript", ".mjs")


class DualRoot(SimpleHTTPRequestHandler):
    def translate_path(self, path: str) -> str:
        clean = path.split("?", 1)[0].split("#", 1)[0]
        if clean.startswith("/corpus/"):
            rel = clean[len("/corpus/"):]
            target = (CORPUS / rel).resolve()
            # Refuse to serve outside the corpus root even if the URL tries.
            if not str(target).startswith(str(CORPUS.resolve())):
                return str(CORPUS)
            return str(target)
        return super().translate_path(path)

    def log_message(self, *args):  # quiet; the harness is noisy enough
        pass

    def end_headers(self):
        # No caching: the harness reloads the same URLs after every rebuild.
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8900)
    args = ap.parse_args()

    os.chdir(PUBLIC)
    handler = functools.partial(DualRoot, directory=str(PUBLIC))
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    print(f"serving {PUBLIC} and /corpus -> {CORPUS} at http://127.0.0.1:{args.port}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
