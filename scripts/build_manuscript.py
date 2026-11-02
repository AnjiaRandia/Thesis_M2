#!/usr/bin/env python3
"""Compile one manuscript using the frozen publication figures and tables."""

from __future__ import annotations
import argparse
import os
from pathlib import Path
import shutil
import subprocess


def run(command: list[str], cwd: Path, environment: dict, log: Path) -> None:
    with log.open("w") as stream:
        subprocess.run(command, cwd=cwd, env=environment, stdout=stream, stderr=subprocess.STDOUT, check=True)


def build(root: Path, language: str) -> None:
    source = root / "manuscript" / language
    output = root / "build" / language
    work = output / "work"
    output.mkdir(parents=True, exist_ok=True)
    if work.exists():
        shutil.rmtree(work)
    shutil.copytree(source, work, symlinks=True)
    (work / "images").unlink()
    (work / "images").symlink_to(root / "figures", target_is_directory=True)
    environment = os.environ.copy()
    engine = shutil.which("pdflatex")
    bibtex = shutil.which("bibtex")
    if not engine or not bibtex:
        raise RuntimeError("The document build requires pdfLaTeX and BibTeX.")
    command = [engine]
    if language == "fr":
        environment["TEXINPUTS"] = str(work / "texmf" / "tex") + "//:" + environment.get("TEXINPUTS", "")
        format_dir = output / "format"
        format_dir.mkdir(exist_ok=True)
        format_path = format_dir / "pdflatex-fr.fmt"
        if not format_path.exists():
            pdftex = shutil.which("pdftex")
            if not pdftex:
                raise RuntimeError("The French format requires pdfTeX.")
            run(
                [pdftex, "-ini", "-etex", "-jobname=pdflatex-fr", "-progname=pdflatex", "pdflatex.ini"],
                format_dir,
                environment,
                output / "format.log",
            )
        command.append("-fmt=" + str(format_path))
    command += ["-interaction=nonstopmode", "-halt-on-error", "MainThesis.tex"]
    run(command, work, environment, output / "latex1.log")
    run([bibtex, "MainThesis"], work, environment, output / "bibtex.log")
    import re

    for index in range(2, 9):
        log_path = output / f"latex{index}.log"
        run(command, work, environment, log_path)
        log = log_path.read_text(errors="replace")
        unsettled = (
            "Label(s) may have changed" in log
            or "Rerun to get outlines right" in log
            or "Rerun to get cross-references right" in log
        )
        if index >= 3 and not unsettled:
            break
    else:
        raise RuntimeError("LaTeX references did not converge after eight passes.")
    if re.search(r"(?:Citation|Reference) .+ undefined|There were undefined references", log):
        raise RuntimeError("Unresolved references; inspect " + str(log_path))
    shutil.copy2(work / "MainThesis.pdf", output / "MainThesis.pdf")
    print(output / "MainThesis.pdf")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", choices=["en", "fr"], required=True)
    arguments = parser.parse_args()
    build(Path(__file__).resolve().parents[1], arguments.language)


if __name__ == "__main__":
    main()
