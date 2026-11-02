#!/usr/bin/env python3
"""Check document content and publication-asset integrity without scientific calculations."""

from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess


def verify(root: Path) -> dict:
    references = json.loads((root / "config/manuscript_reference.json").read_text())
    with (root / "config/figure_manifest.csv").open() as stream:
        figures = list(csv.DictReader(stream))
    for row in figures:
        path = root / row["path"]
        assert path.stat().st_size == int(row["bytes"]), path
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"], path
    results = {}
    for language, reference in references.items():
        pdf = root / "build" / language / "MainThesis.pdf"
        text = subprocess.check_output(["pdftotext", str(pdf), "-"], stderr=subprocess.DEVNULL).decode()
        info = subprocess.check_output(["pdfinfo", str(pdf)]).decode()
        pages = int(re.search(r"Pages:\s*(\d+)", info).group(1))
        digest = hashlib.sha256(re.sub(r"\s+", " ", text).strip().encode()).hexdigest()
        assert pages == reference["pages"], (language, pages, reference["pages"])
        assert digest == reference["text_sha256"], (language, "document text differs")
        assert all(page.strip() for page in text.split("\f")[:-1]), (language, "empty page")
        results[language] = dict(pages=pages, text_matches_reference=True, blank_pages=False)
    return dict(
        documents=results,
        verified_figure_assets=len(figures),
        scientific_calculations_run=False,
    )


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    report = verify(root)
    path = root / "build/verification.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
