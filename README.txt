KINEMATICALLY COLD AND WARM ATOMIC HYDROGEN IN NEARBY GALAXIES

This repository contains the final English and French thesis sources, the
publication figures and the scientific methods retained in the thesis.
It covers the 16-galaxy sample, local kinematic classification, spatial
comparisons, Kennicutt–Schmidt relations and star formation efficiencies.

DOCUMENT BUILD

From this directory:

    snakemake --cores 2

Outputs:
    build/en/MainThesis.pdf
    build/fr/MainThesis.pdf
    build/verification.json

This default target compiles the documents using the existing publication
figures and numerical tables. It does not run ROHSA, rebuild maps, fit gas–SFR
relations, bootstrap measurements or rerun the scientific analyses. It needs
Python/Snakemake, pdfLaTeX, BibTeX, pdfTeX and Poppler (pdftotext/pdfinfo).
French language support and hyphenation patterns are provided locally.
The document build also works without data/raw, data/support or cache; only
the manuscript sources, publication assets and document tools are required.
LaTeX runs until references settle. Verification compares the extracted text
and page counts against the final manuscripts and checks every figure checksum.

Useful targets:
    snakemake --cores 1 english
    snakemake --cores 1 french
    snakemake --cores 2 latex
    snakemake --cores 2 --dry-run
    snakemake --cores 1 data_inventory  # Optional: check locally restored data

A second invocation with unchanged inputs should report nothing to be done.

LAYOUT

    manuscript/en/          Final English source
    manuscript/fr/          Final French source
    figures/                Only assets included by the current manuscripts
    config/                 Retained sample, ROHSA settings and file manifests
    data/raw/               Original survey inputs, linked locally
    data/support/           Source masks used by native-cube preparation
    cache/                  Frozen fitted and matched scientific inputs
    src/hiphases/           Retained numerical and plotting functions
    scripts/                Document builder and explicit scientific commands
    build/                  Generated documents, logs and verification
    results/                Explicitly regenerated scientific products

DATA AND STORAGE

No original observational data were copied. The 54 selected raw files are
linked under data/raw; their paths and sizes are in config/data_manifest.csv.
They occupy approximately 6.11 GB. The 16 derived SoFiA masks are kept separately
under data/support. Frozen products are listed in config/product_manifest.csv.
These local data, support and cache directories are ignored by Git. They can
later be distributed as release assets with a download/checksum manifest.

The 40 included publication assets are ordinary files that share disk storage
with the original figures through hard links on this workstation. The manuscript
image directories link to figures/ within this repository, so the tracked source
and publication figures have no dependency on the old directory layout.

SCIENTIFIC COMMANDS

These commands are explicit and are not dependencies of the default document
build. They write to results/, never into the frozen cache or manuscripts.
The analysis environment is described in environment.yml. ROHSA also requires
its separately installed executable, selected with HIPHASES_ROHSA.

    python scripts/decompose.py prepare NGC5194
    python scripts/decompose.py fit NGC5194
    python scripts/decompose.py classify NGC5194
    python scripts/kennicutt_schmidt.py NGC5194
    python scripts/kennicutt_schmidt.py --combined
    python scripts/star_formation_efficiency.py
    python scripts/spiral_arms.py --stage prepare
    python scripts/spiral_arms.py --stage analyse
    python scripts/match_molecular_maps.py
    python scripts/molecular_association.py

ROHSA uses the adopted amplitude/global-width penalties of zero, local centroid
and dispersion penalties of one, per-galaxy Gaussian counts and a three-channel
minimum FWHM. The sampled gas–SFR estimator is the density-weighted orthogonal
regression with paired bootstrap intervals. Obsolete Bayesian/bisector fitting,
threshold scans, stellar-environment weighting and exploratory diagnostics are
not included. The independent original survey inputs and the frozen scientific
products remain distinct: the document-only build verifies the publication
snapshot; it is not a validation of a new raw-data-to-ROHSA run.

The fitted native parameters and inherited matched total-HI/SFR/CO maps are
retained together because the final phase fractions were applied to those
matched total-HI maps. Regenerating a different total-HI map would change that
input convention and must not silently replace the frozen publication products.

NOTES

Markdown files (*.md) are intentionally ignored by Git, as are local path maps,
intermediate products, old backups, build files and scientific caches. README.txt
is the public entry point. No remote repository or public release is configured.
The workflow and both manuscript builds were checked without rerunning ROHSA or
any scientific analysis.
