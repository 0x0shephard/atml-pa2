# NeurIPS formatting template

`main.tex` and `main.pdf` are the completed **AI-assisted sample report**, requested for an ungraded sample assignment. All measurements come from saved experiments. The report explicitly discloses that the human safety audit was not performed; no manual labels or agreement scores were fabricated. `template.tex` preserves the earlier blank writing template.

`neurips_2026.sty` is the unmodified official style from https://media.neurips.cc/Conferences/NeurIPS2026/Formatting_Instructions_For_NeurIPS_2026.zip. It is used with `preprint,nonatbib`; margins, spacing, and font sizes are not overridden. If your instructor provides a specific course/year template, use that instead.

The compiled report contains eight main-content pages and a ninth references page. The course's eight-page rule takes precedence over the conference's own paper length rule. The graded manual still requires student-written report language and a completed blind audit; the sample does not claim to satisfy those requirements.

Build:

```bash
python -m scripts.report_tables
python -m scripts.report_analysis
cd report
tectonic main.tex
```

The abstract includes https://github.com/0x0shephard/atml-pa2. Primary method and benchmark sources appear in the bibliography. The manual-audit interface remains `../submission/manual_audit.html`; its CSV export must replace `../results/task4_safety/manual_audit.csv` before human agreement can be computed.

Training-only times in the resource table come from standard-run provenance. End-to-end Kaggle job times are separately stored in `../results/jobs/`.
