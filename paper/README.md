# LAMP: ACM manuscript sources

This package uses the official ACM `acmart` class with `manuscript,screen,nonacm` options. It is an independently distributed technical report; ACM publication or acceptance is not claimed.

Files: `main.tex`, `references.bib`, `main.bbl`, `LAMP.pdf`, and `TECHNICAL_PAPER.md`.

Build using a TeX distribution with acmart:

```sh
latexmk -pdf main.tex
```

Or Tectonic (tested with 0.17.0 and bundled acmart v1.83):

```sh
tectonic --keep-logs --keep-intermediates main.tex
```

The build produces `main.pdf`; the distribution copy is named `LAMP.pdf`. All equations and references are editable. The Markdown version retains the same technical content. Reference metadata was transcribed from the original report; this format conversion does not constitute a new literature review or experimental validation.

Original project: https://github.com/FL2744/LAMP
