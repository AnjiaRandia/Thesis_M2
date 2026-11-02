"""Publication build using existing figures and fitted results."""
from pathlib import Path
import csv
import sys

configfile: "config/workflow.yaml"
PYTHON = sys.executable
LANGUAGES = config["languages"]
with open("config/figure_manifest.csv") as stream:
    FIGURES = [row["path"] for row in csv.DictReader(stream)]


def manuscript_sources(wildcards):
    base = Path("manuscript") / wildcards.language
    extensions = {".tex", ".cls", ".sty", ".bib", ".bst", ".ldf", ".lua", ".ini", ".pat", ".lic", ".txt"}
    return sorted(str(path) for path in base.rglob("*")
                  if path.is_file() and path.suffix in extensions)


rule all:
    input:
        "build/verification.json"


rule latex:
    input:
        expand("build/{language}/MainThesis.pdf", language=LANGUAGES)


rule english:
    input:
        "build/en/MainThesis.pdf"


rule french:
    input:
        "build/fr/MainThesis.pdf"


rule manuscript:
    input:
        sources=manuscript_sources,
        figures=FIGURES,
        builder="scripts/build_manuscript.py"
    output:
        "build/{language}/MainThesis.pdf"
    wildcard_constraints:
        language="en|fr"
    threads: 1
    resources:
        mem_mb=1024
    shell:
        "{PYTHON:q} scripts/build_manuscript.py --language {wildcards.language:q}"


rule verify_documents:
    input:
        pdfs=expand("build/{language}/MainThesis.pdf", language=LANGUAGES),
        reference="config/manuscript_reference.json",
        figures="config/figure_manifest.csv",
        verifier="scripts/verify_documents.py"
    output:
        "build/verification.json"
    shell:
        "{PYTHON:q} scripts/verify_documents.py"


rule data_inventory:
    input:
        "config/data_manifest.csv",
        "config/product_manifest.csv",
        "scripts/verify_inputs.py"
    output:
        "build/data_inventory.json"
    shell:
        "{PYTHON:q} scripts/verify_inputs.py"
