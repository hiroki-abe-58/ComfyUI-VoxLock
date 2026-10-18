"""The installed package must not contain code that reads environment variables, opens network connections or starts processes (the
categories of the Comfy Registry's automated scan; scripts/ and tests/ are not part of a Registry package), and must never let the
voxlock package download models at run time."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_FILES = [p for p in [ROOT / "__init__.py", *(ROOT / "voxlock_comfy").rglob("*.py"), *(ROOT / "tools").rglob("*.py")]
                 if "__pycache__" not in p.parts]
FORBIDDEN = re.compile(r"os\.environ|getenv|putenv|urllib|requests\.|http\.client|socket\.|subprocess|os\.system|importlib|(?<![.\w])exec\(|(?<![.\w])eval\(|__import__")


def test_package_has_no_environment_network_or_process_code():
    hits = [f"{p.relative_to(ROOT).as_posix()}:{n}: {line.strip()}" for p in PACKAGE_FILES
            for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1) if FORBIDDEN.search(line)]
    assert not hits, "\n".join(hits)


def test_the_nodes_never_use_the_package_downloader():
    """VoxLock(model_dir=None) and voxlock.core.download() fetch from Hugging Face; the nodes always pass a local folder."""
    text = "\n".join(p.read_text(encoding="utf-8") for p in (ROOT / "voxlock_comfy").rglob("*.py"))
    assert "download(" not in text and "hf_hub" not in text
    assert "VoxLock(model_dir=model_dir" in text
