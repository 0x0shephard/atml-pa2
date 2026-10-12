# NeurIPS formatting template

This is a formatting and numerical-evidence template, **not a completed report**. The assignment prohibits AI-written report language, interpretation, and analysis. Replace every gray student-writing placeholder, write all captions yourself, and complete the manual audit before submitting.

`neurips_2026.sty` is the unmodified official style from https://media.neurips.cc/Conferences/NeurIPS2026/Formatting_Instructions_For_NeurIPS_2026.zip. It is used with `preprint,nonatbib`; margins, spacing, and font sizes are not overridden. If your instructor provides a specific course/year template, use that instead.

The template reserves eight main-content pages and puts references on a separate page. The course's eight-page rule takes precedence over the conference's own paper length rule. Reflow your completed writing naturally and check the final page count; do not submit placeholder or blank pages.

Build:

```bash
python -m scripts.report_tables
cd report
tectonic main.tex
```

The abstract includes https://github.com/0x0shephard/atml-pa2. Enter your own title, student ID, and institution. Replace the placeholder reference section with sources you actually read and cited. The manual-audit interface is `../submission/manual_audit.html`; its CSV export must replace `../results/task4_safety/manual_audit.csv` before safety agreement is recomputed.

Training-only times in the resource table come from standard-run provenance. End-to-end Kaggle job times are separately stored in `../results/jobs/`.
