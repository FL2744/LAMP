# LAMP illustrated technical report

**Technical paper:** [PDF](main.pdf) · [LaTeX](main.tex) · [Markdown](TECHNICAL_PAPER.md) · [Illustrations](figures/)

Five tightly cropped, reproducible technical diagrams explain the mechanisms described in the report. These are explanatory schematics, not application screenshots or new experimental results. The existing literature and scientific qualifications are retained. AI disclosure precedes references.

Build with `tectonic main.tex` (rename the resulting PDF to `main.pdf` where needed). The source archive includes all figure PDFs and PNGs.

Regenerate the diagrams with `python3 render_figures.py` (requires Matplotlib). Diagram titles and captions are recorded in `ILLUSTRATIONS.json`; vector PDF and raster PNG versions are provided.
