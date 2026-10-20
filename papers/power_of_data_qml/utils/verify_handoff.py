"""Read-only handoff verification: recompute curated artifacts and cross-check documents.

Usage (from the paper folder)::

    python utils/verify_handoff.py

Writes nothing to the repository. Generated files go to a temporary directory
that is deleted on exit. Checks:

1. the convention-sweep candidate table, the system comparison and the photonic
   scaling report, recomputed from the raw run metrics, are byte-identical to
   the files in ``results/``;
2. the three study aggregates are COMPLETE with the documented record counts;
3. headline values computed from the aggregates appear, with the same rounding,
   in README.md, CONFLUENCE.md and notebook.ipynb.

Exits non-zero on the first mismatch.
"""

from __future__ import annotations

import filecmp
import json
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_DIR / "results"
OUTDIR = PROJECT_DIR / "outdir"

RECOMPUTE = [
    (
        "report_convention_check.py",
        [
            "--run", str(OUTDIR / "run_20261017-033107"), "--stage", "A (attempt 2)",
            "--run", str(OUTDIR / "run_20261016-163902"), "--stage", "A2",
        ],
        "convention_check_candidates",
        ".md",
    ),
    (
        "compare_systems.py",
        [
            "--gate", str(RESULTS / "study_extended.json"),
            "--photonic", str(RESULTS / "study_photonic_extended.json"),
            "--photonic-capped", str(RESULTS / "study_photonic_capped.json"),
            "--training-size", str(RESULTS / "training_size.json"),
        ],
        "comparison_extended",
        ".json",
    ),
    (
        "report_photonic_scaling.py",
        [
            "--probe", str(OUTDIR / "run_20261019-160710"),
            "--study-run", str(OUTDIR / "run_20261017-141820"),
            "--study-run", str(OUTDIR / "run_20261019-164643"),
            "--study-run", str(OUTDIR / "run_20261019-212612"),
            "--study-run", str(OUTDIR / "run_20261019-231148"),
        ],
        "photonic_scaling",
        ".json",
    ),
]  # fmt: skip
AGGREGATES = {
    "study_extended": 640,
    "study_photonic_extended": 440,
    "study_photonic_capped": 120,
}


def summary_mean(name, component, quantity, embedding, n, protocol, train_size):
    rows = json.loads((RESULTS / f"{name}.json").read_text())["summary"]
    for r in rows:
        if (r["component"], r["quantity"], r["embedding"], r["n"], r["protocol"], r["train_size"]) == (
            component, quantity, embedding, n, protocol, train_size,
        ):  # fmt: skip
            return r["mean"]
    raise KeyError((name, component, quantity, embedding, n))


def main() -> int:
    failures = []
    with tempfile.TemporaryDirectory() as scratch:
        for script, arguments, stem, suffix in RECOMPUTE:
            target = Path(scratch) / stem
            subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_DIR / "utils" / script),
                    *arguments,
                    "--output",
                    str(target),
                ],
                check=True,
                capture_output=True,
                cwd=PROJECT_DIR,
            )
            same = filecmp.cmp(
                target.with_suffix(suffix), RESULTS / f"{stem}{suffix}", shallow=False
            )
            print(
                f"[{'ok' if same else 'MISMATCH'}] recomputed {stem}{suffix} is byte-identical"
            )
            if not same:
                failures.append(stem)
    for name, expected in AGGREGATES.items():
        payload = json.loads((RESULTS / f"{name}.json").read_text())
        ok = (
            payload["coverage"] == "COMPLETE"
            and payload["observed_records"] == expected
            and not payload["missing"]
        )
        print(
            f"[{'ok' if ok else 'MISMATCH'}] {name}: {payload['coverage']} {payload['observed_records']}/{expected}"
        )
        if not ok:
            failures.append(name)

    pqk = "projected_quantum_kernel_minus_best_classical"
    headline = {
        "gate E3 n=20 PQK-best": f"+{summary_mean('study_extended', 'engineered_pqk', pqk, 'E3', 20, 'cv', 600):.3f}",
        "gate E2 n=20 PQK-best": f"+{summary_mean('study_extended', 'engineered_pqk', pqk, 'E2', 20, 'cv', 600):.3f}",
        "photonic PB n=16 PQK-best": f"+{summary_mean('study_photonic_extended', 'engineered_pqk', pqk, 'PB', 16, 'cv', 600):.3f}",
        "capped PB n=20 PQK-best": f"+{summary_mean('study_photonic_capped', 'engineered_pqk', pqk, 'PB', 20, 'cv', 600):.3f}",
        "gate E3 n=20 d_QK": f"{summary_mean('study_extended', 'dimension_geometry', 'd_quantum_kernel', 'E3', 20, None, None):.1f}",
    }  # fmt: skip
    documents = {
        name: (PROJECT_DIR / name).read_text()
        for name in ("README.md", "CONFLUENCE.md", "notebook.ipynb")
    }
    for label, value in headline.items():
        for document, text in documents.items():
            # CONFLUENCE and the notebook may round to two decimals; README carries three.
            short = (
                value[:-1]
                if value.count(".") and len(value.split(".")[1]) == 3
                else value
            )
            whole = f"{round(float(value))}"  # e.g. "795 of 800" for d = 794.7
            found = value in text or (
                document != "README.md"
                and (
                    short in text
                    or (label.endswith("d_QK") and f"{whole} of 800" in text)
                )
            )
            print(f"[{'ok' if found else 'MISSING'}] {label} = {value} in {document}")
            if not found:
                failures.append(f"{label}:{document}")
    print("RESULT:", "PASS" if not failures else f"FAIL {failures}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
