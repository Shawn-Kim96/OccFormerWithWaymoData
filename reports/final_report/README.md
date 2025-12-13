# Final Report (LaTeX)

Main file: `final_report/main.tex`

Figures used by the report are under `final_report/figures/`.

## Build
If you have LaTeX installed:
```bash
cd final_report
pdflatex main.tex
pdflatex main.tex
```

If you do not have LaTeX locally, the easiest option is to upload `main.tex` and the `figures/` folder to Overleaf.

## Update figures (optional)
The figures were generated from `results/*/logs/evaluate.log`. To regenerate:
```bash
python final_report/make_figures.py
```
