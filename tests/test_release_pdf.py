"""Release gate: no "TODO" or "PENDING" in the text of the compiled PDFs (the data-availability
placeholders \\dataurl in main.tex and titlepage.tex, and any \\todo). Run with --release after
building the PDFs (paper/release_checklist.md)."""
import pathlib
import shutil
import subprocess

import pytest

PAPER = pathlib.Path(__file__).resolve().parent.parent / "paper"
PDFS = ["main.pdf", "supplementary.pdf", "titlepage.pdf"]


@pytest.mark.release
@pytest.mark.parametrize("name", PDFS)
def test_pdf_has_no_todo_or_pending(name):
    exe = shutil.which("pdftotext")
    assert exe, "pdftotext not found on PATH"
    pdf = PAPER / name
    assert pdf.exists(), f"{pdf} not built"
    text = subprocess.run([exe, "-layout", str(pdf), "-"], capture_output=True, check=True).stdout.decode("utf-8", "ignore")
    hits = [w for w in ("TODO", "PENDING") if w in text]
    assert not hits, f"{name} still contains {hits}"
