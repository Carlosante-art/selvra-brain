"""Cross-platform browser-opening helper.

WSL2 har inte default file:// association — webbrowser.open() failar
silent. Vi detekterar WSL och använder explorer.exe (öppnar i Windows-
default-browser) istället.

Strategier (i ordning):
1. WSL → explorer.exe FILE
2. WSL → wslview FILE
3. macOS → open
4. Linux native → webbrowser.open
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import webbrowser
from pathlib import Path


def is_wsl() -> bool:
    """True om vi kör i Windows Subsystem for Linux."""
    try:
        release = platform.uname().release.lower()
        return "microsoft" in release or "wsl" in release
    except Exception:
        return False


def open_html(path: Path | str) -> tuple[bool, str]:
    """Försök öppna en HTML-fil i browsern.

    Returnerar (success, method) — `method` är str-namn på vad vi
    försökte (för debug-output).
    """
    p = Path(path).resolve()

    if is_wsl():
        # explorer.exe är vanligast på WSL — finns även i minimal Ubuntu
        if shutil.which("explorer.exe"):
            try:
                subprocess.Popen(
                    ["explorer.exe", str(p)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                return True, "explorer.exe"
            except Exception:  # noqa: BLE001
                pass
        # wslview från wslu-paketet
        if shutil.which("wslview"):
            try:
                subprocess.Popen(
                    ["wslview", str(p)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                return True, "wslview"
            except Exception:  # noqa: BLE001
                pass

    # macOS
    if platform.system() == "Darwin":
        try:
            subprocess.Popen(["open", str(p)])
            return True, "open"
        except Exception:  # noqa: BLE001
            pass

    # Linux native / fallback
    url = f"file://{p}"
    if webbrowser.open(url):
        return True, "webbrowser"

    return False, "manual"


def print_open_instruction(path: Path | str) -> None:
    """Skriv ut tydlig instruktion för manuell open."""
    p = Path(path).resolve()
    print(f"Open in browser: file://{p}")
    if is_wsl():
        print(f"  Or run:        explorer.exe {p}")
