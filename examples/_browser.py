"""Cross-platform browser-opening + lokal HTTP-server för viz.

Två problem som denna modul löser:

1. WSL2 har inte default file:// association — webbrowser.open() failar
   silent. Vi detekterar WSL och använder explorer.exe (öppnar i Windows-
   default-browser) istället.

2. Moderna browsers blockerar fetch() från file:// — CORS gör att
   index.html inte kan polla state.json. Lösningen är en lokal HTTP-
   server som serverar viz/-mappen.

Användning från examples:

    from examples._browser import open_viz_in_browser

    server = open_viz_in_browser(VIZ_DIR, port=8765)
    # Server kör i bakgrundstråd. Demon fortsätter.
"""

from __future__ import annotations

import http.server
import platform
import shutil
import socket
import socketserver
import subprocess
import threading
import webbrowser
from functools import partial
from pathlib import Path


def is_wsl() -> bool:
    """True om vi kör i Windows Subsystem for Linux."""
    try:
        release = platform.uname().release.lower()
        return "microsoft" in release or "wsl" in release
    except Exception:
        return False


def _find_free_port(preferred: int = 8765) -> int:
    """Hitta en ledig port — preferred om möjligt, annars OS-tilldelad."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        try:
            s.bind(("127.0.0.1", preferred))
            return preferred
        except OSError:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]
    finally:
        s.close()


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    """SimpleHTTPRequestHandler utan request-logging-spam."""

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        # Tystare — vi vill inte att varje state.json-poll spammar terminalen
        pass


def start_viz_server(
    viz_dir: Path,
    *,
    port: int = 8765,
) -> tuple[socketserver.TCPServer, int]:
    """Starta lokal HTTP-server för viz/ i bakgrundstråd.

    Servern serverar viz/-mappen. Browsers kan då polla state.json
    via http://localhost:PORT/state.json utan CORS-problem.

    Returnerar (server, actual_port). Daemon-tråd → dör med processen.
    """
    actual_port = _find_free_port(port)
    handler = partial(_QuietHandler, directory=str(viz_dir.resolve()))
    server = socketserver.TCPServer(("127.0.0.1", actual_port), handler)
    server.allow_reuse_address = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, actual_port


def open_url(url: str) -> tuple[bool, str]:
    """Försök öppna en URL i browsern.

    WSL → explorer.exe (öppnar i Windows-default-browser)
    macOS → open
    Linux native → webbrowser.open
    """
    if is_wsl():
        if shutil.which("explorer.exe"):
            try:
                subprocess.Popen(
                    ["explorer.exe", url],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                return True, "explorer.exe"
            except Exception:  # noqa: BLE001
                pass
        if shutil.which("wslview"):
            try:
                subprocess.Popen(
                    ["wslview", url],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                return True, "wslview"
            except Exception:  # noqa: BLE001
                pass

    if platform.system() == "Darwin":
        try:
            subprocess.Popen(["open", url])
            return True, "open"
        except Exception:  # noqa: BLE001
            pass

    if webbrowser.open(url):
        return True, "webbrowser"

    return False, "manual"


def open_viz_in_browser(
    viz_dir: Path,
    *,
    port: int = 8765,
    open_browser: bool = True,
) -> socketserver.TCPServer:
    """Convenience: starta server + öppna browser till localhost.

    Detta är vad alla examples ska kalla. Returnerar server-objekt
    (daemon-tråd) — caller behöver inte stoppa, dör med processen.
    """
    server, actual_port = start_viz_server(viz_dir, port=port)
    url = f"http://localhost:{actual_port}/index.html"
    print(f"Viz server: {url}")

    if open_browser:
        ok, method = open_url(url)
        if ok:
            print(f"  Opening with {method}")
        else:
            print(f"  Open manually: {url}")
            if is_wsl():
                print(f"  Or: explorer.exe {url}")

    return server


# ─── Legacy file:// helpers (kvar för bakåtkomp) ────────────────────


def open_html(path: Path | str) -> tuple[bool, str]:
    """Legacy: försök öppna en HTML-fil via file:// URL.

    OBS: fungerar inte för viz/ — moderna browsers blockerar fetch()
    från file:// vilket bryter state-polling. Använd open_viz_in_browser()
    istället för viz/.
    """
    p = Path(path).resolve()
    return open_url(f"file://{p}")


def print_open_instruction(path: Path | str) -> None:
    """Skriv ut tydlig instruktion för manuell open."""
    p = Path(path).resolve()
    print(f"Open in browser: file://{p}")
    if is_wsl():
        print(f"  Or run:        explorer.exe {p}")
