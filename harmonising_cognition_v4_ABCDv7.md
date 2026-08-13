# harmonising_cognition_v4_ABCDv7.py

Harmonises raw cognitive data from the ABCD Study (v7) across five domains, producing z-scored variables at four levels of standardisation. Output is one row per participant × session.

---

## Input files

All input files are tab-separated (`.tsv`) located in the same directory as the script, or passed via `--data-dir`.

| File | Content |
|---|---|
| `ab_g_dyn.tsv` | Design variables including site (`ab_g_dyn__design_site`) |
| `nc_y_wisc.tsv` | WISC-V Matrix Reasoning (IQ) |
| `nc_y_nihtb.tsv` | NIH Toolbox: Flanker, DCCS/Card Sort, List Sorting Working Memory |
| `nc_y_flnkr.tsv` | Millisecond Flanker (response inhibition fallback) |
| `nc_y_lmt.tsv` | Little Man Task (attention) |

**Key identifier variables** (consistent across all files):
- `participant_id` — participant identifier
- `session_id` — wave (`ses-00A`, `ses-02A`, `ses-04A`, `ses-06A`)
- `ab_g_dyn__design_site` — site number (1–21)

---

## Output file

`harmonised_cognition_v4_ABCDv7.csv` — one row per participant × session (up to 4 rows per participant). Missing data is represented as `n/a`.

**Shape:** ~28,000 rows × 35 columns (full ABCD cohort; scales from toy dataset)

---

## Z-scoring variants

Four z-scoring levels are computed for each cognitive domain:

| Suffix | Grouping | Description |
|---|---|---|
| `_z_withinsite` | Site | Z-scored separately within each of the 21 ABCD sites |
| `_z_withinsitewave` | Site × Wave | Z-scored within each site × timepoint combination |
| `_z_withincohort` | None (full cohort) | Z-scored across all sites treated as one cohort |
| `_z_withincohortwave` | Wave | Z-scored across all sites but separately within each wave |

**Edge case handling:** groups with sample size ≤ 1 or SD = 0 are assigned a z-score of 0 (the group mean) rather than `NaN`. This affects very small site × wave cells, not the full-cohort z-scores.

---

## Cognitive domains

### 1. IQ — WISC-V Matrix Reasoning (`nc_y_wisc.tsv`)

Higher score = better performance. No directionality flip applied.

**Source columns:**
- `nc_y_wisc__raw_score` → `IQ_raw_notz`
- `nc_y_wisc__scaled_score` → `IQ_scaled_notz`

| Output variable | Description |
|---|---|
| `IQ_raw_notz` | Raw score (number of items correct) |
| `IQ_scaled_notz` | Scaled score (normative scale: M=10, SD=3) |
| `IQ_raw_z_withinsite` | Raw score z-scored within site |
| `IQ_scaled_z_withinsite` | Scaled score z-scored within site |
| `IQ_raw_z_withinsitewave` | Raw score z-scored within site × wave |
| `IQ_scaled_z_withinsitewave` | Scaled score z-scored within site × wave |
| `IQ_raw_z_withincohort` | Raw score z-scored across full cohort |
| `IQ_scaled_z_withincohort` | Scaled score z-scored across full cohort |
| `IQ_raw_z_withincohortwave` | Raw score z-scored across cohort within each wave |
| `IQ_scaled_z_withincohortwave` | Scaled score z-scored across cohort within each wave |

---

### 2. Response Inhibition — NIH Toolbox Flanker / Millisecond Flanker

**Task selection rule:** NIH Toolbox Flanker (`nihtb_flanker`) is used when the computed score is available. Millisecond Flanker (`millisecond_flanker`) is the fallback when NIH Toolbox is unavailable at that session. The `task_type` column records which task was used for each row.

**NIH Toolbox Flanker — scoring algorithm (`nc_y_nihtb__flnkr__computed_score`):**

The NIH Toolbox uses a validated two-stage algorithm based on incongruent trial performance:
- If incongruent accuracy ≤ 80%: score is accuracy-based (performance differences driven by accuracy)
- If incongruent accuracy > 80%: score is RT-cost-based (congruent RT − incongruent RT), because accuracy is at ceiling and can no longer discriminate

Higher computed score = better inhibitory control. No directionality flip applied.

**Millisecond Flanker — incongruent accuracy (`nc_y_flnkr__incongr_acc`, v4):**

Incongruent trial accuracy is used rather than overall accuracy because:
1. Congruent trials do not test response inhibition — the flankers point the same way as the target and impose no conflict
2. Incongruent accuracy directly indexes the ability to suppress the prepotent conflicting response, which is the construct being measured
3. This is consistent with what the NIH Toolbox Flanker computed_score is fundamentally built around

Higher incongruent accuracy = better conflict resolution. No directionality flip applied.

**Z-scoring note:** z-scoring is computed separately within each task type before scores are combined into a single column, regardless of the grouping level. This prevents the two tasks' different scales from contaminating each other.

**Source columns:**
- `nc_y_nihtb__flnkr__computed_score` → `RespInhib_computed_notz`
- `nc_y_flnkr__incongr_acc` → `RespInhib_incongr_acc_notz`

| Output variable | Description |
|---|---|
| `RespInhib_computed_notz` | NIH Toolbox Flanker computed score (raw; nihtb rows only) |
| `RespInhib_incongr_acc_notz` | Millisecond Flanker incongruent accuracy (raw; millisecond rows only) |
| `RespInhib_z_withinsite` | Response inhibition z-score within site |
| `RespInhib_z_withinsitewave` | Response inhibition z-score within site × wave |
| `RespInhib_z_withincohort` | Response inhibition z-score across full cohort |
| `RespInhib_z_withincohortwave` | Response inhibition z-score across cohort within each wave |
| `task_type` | Task used: `nihtb_flanker`, `millisecond_flanker`, or `n/a` |

---

### 3. Cognitive Flexibility — NIH Toolbox DCCS / Card Sort (`nc_y_nihtb.tsv`)

The NIH Toolbox Dimensional Change Card Sort (DCCS) is identified in ABCD data as `crdst` (Card Sort). Higher computed score = better flexible switching. No directionality flip applied.

**Source column:** `nc_y_nihtb__crdst__computed_score` → `CogFlex_computed_notz`

| Output variable | Description |
|---|---|
| `CogFlex_computed_notz` | DCCS computed score (raw) |
| `CogFlex_z_withinsite` | Z-scored within site |
| `CogFlex_z_withinsitewave` | Z-scored within site × wave |
| `CogFlex_z_withincohort` | Z-scored across full cohort |
| `CogFlex_z_withincohortwave` | Z-scored across cohort within each wave |

---

### 4. Working Memory — NIH Toolbox List Sorting (`nc_y_nihtb.tsv`)

Higher raw score = more correct reorderings = better working memory. No directionality flip applied.

**Source column:** `nc_y_nihtb__lswmt__raw_score` → `WorkingMem_raw_notz`

| Output variable | Description |
|---|---|
| `WorkingMem_raw_notz` | List Sorting raw score |
| `WorkingMem_z_withinsite` | Z-scored within site |
| `WorkingMem_z_withinsitewave` | Z-scored within site × wave |
| `WorkingMem_z_withincohort` | Z-scored across full cohort |
| `WorkingMem_z_withincohortwave` | Z-scored across cohort within each wave |

---

### 5. Attention — Little Man Task (`nc_y_lmt.tsv`)

Higher accuracy = better attentional performance. No directionality flip applied.

**Source column:** `nc_y_lmt__crct_acc` → `Attention_raw_notz`

**Data quality note:** A small number of rows in the ABCD v7 data have `nc_y_lmt__crct_acc` values > 1 (e.g. 90.625, 93.75, 96.875, 100). These are percentage-coded entries (should be 0–1) and are set to `n/a` before z-scoring. The valid range is 0–1.

| Output variable | Description |
|---|---|
| `Attention_raw_notz` | Proportion of trials correct (0–1; rows with values > 1 excluded as coding errors) |
| `Attention_z_withinsite` | Z-scored within site |
| `Attention_z_withinsitewave` | Z-scored within site × wave |
| `Attention_z_withincohort` | Z-scored across full cohort |
| `Attention_z_withincohortwave` | Z-scored across cohort within each wave |

---

## Usage

```bash
# Local (input files and output in same directory as script)
python harmonising_cognition_v4_ABCDv7.py

# Custom paths (e.g. on KCL CREATE TRE HPC)
python3 harmonising_cognition_v4_ABCDv7.py \
    --data-dir /dataset/abcd/v7/phenotype \
    --output   /job_scratch/harmonised_cognition_v4_ABCDv7.csv
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--data-dir` | Directory containing the script | Directory with all input TSV files |
| `--output` | `harmonised_cognition_v4_ABCDv7.csv` in `--data-dir` | Full path for the output CSV |
| `--skip-qc` | off | Skip the post-harmonisation QC report and figures |
| `--qc-dir` | A `QC/` subfolder next to `--output` | Directory for QC figures |

> **TRE note:** `--qc-dir` defaults to a folder next to `--output`, *not* inside `--data-dir`. On the KCL CREATE TRE `--data-dir` is `/dataset/abcd/v7/phenotype`, which is **read-only** — writing QC figures there raises `PermissionError` and exits with code 1 after the CSV has already saved. Since `--output` points at `/job_scratch/`, the default resolves to `/job_scratch/QC/` and works without any extra flag.

---

## Quality control

After saving the CSV the script runs a 14-check QC report and writes three figures. Every check prints `PASS` / `WARN` / `FAIL`, with a summary count at the end, so a bad run is visible in a Slurm log without opening the CSV. QC runs on a numeric copy of the table taken *before* `NaN` is replaced with the string `"n/a"`, so the checks aren't confounded by dtype coercion.

| # | Check | What it catches |
|---|---|---|
| 1 | **Row uniqueness** (`participant_id` × `session_id`) | Duplicate rows — these inflate n, distort every group mean/SD, and multiply rows on merge |
| 2 | **Non-missing counts per domain × wave** | A domain unexpectedly empty at a wave, usually a variable/column-name mismatch |
| 3 | **Raw plausible-range check** | Impossible values — this is the check that formalises the Little Man Task percentage-coding error |
| 4 | **Z-score coverage matches raw coverage** | Z-scoring dropping or inventing observations |
| 5 | **Z-score standardisation** | Each variant must be mean 0 / SD 1 within its own grouping. Per-site output lists failures only; per-wave lists all |
| 6 | **Direction / sign check** | Spearman ρ(raw, z) must be exactly **+1** for every ABCD domain — no domain here is sign-flipped. Anything else is a sign error that would silently reverse a domain's meaning downstream |
| 7 | **Z-scoring group sizes** | Site × wave cells with n < 30, where the group mean/SD are themselves noisy |
| 8 | **Distribution shape** | Skew/kurtosis. Z-scoring standardises location and scale but does **not** normalise shape |
| 9 | **Floor / ceiling effects** | % of observations sitting exactly on a scale bound — z-scoring can't fix a pile-up |
| 10 | **Extreme values after z-scoring** | Observations beyond \|z\| > 4 |
| 11 | **Age association** (construct validity) | Each domain should move with age in the expected direction; flat or reversed suggests the variable isn't measuring what we think |
| 12 | **Cross-domain correlations** | All z-scores are sign-aligned "higher = better", so domains should correlate *positively* |
| 13 | **Test–retest consistency** | Within-person correlation between adjacent waves. Near-zero = mostly noise |
| 14 | **Attrition / selection** | Compares baseline scores of participants who do vs don't return — cognition-related dropout biases longitudinal estimates |

**QC figures** (written to `--qc-dir`):

- **`QC_distributions_ABCDv7.png`** — per domain: raw, `_z_withincohort`, `_z_withincohortwave` histograms
- **`QC_centring_monotonicity_ABCDv7.png`** — per-wave boxplots of both z variants, plus a raw-vs-z scatter confirming the map is monotonic and correctly signed
- **`QC_structure_ABCDv7.png`** — cross-domain correlation heatmap, floor/ceiling bars, test–retest correlations

### Plausible-range values

The check-3 ranges were corrected after inspecting the real data:

| Domain | Range | Note |
|---|---|---|
| IQ scaled | (1, 19) | WISC-V scaled score, M=10 SD=3 |
| IQ raw | (0, 30) | Matrix Reasoning raw |
| RespInhib computed | (0, 10) | **Corrected from (0, 150)** — ABCD v7 reports the Flanker computed score on a 0–10 scale, not the 0–150 uncorrected-standard-score scale used elsewhere in NIH Toolbox. The original range was wide enough to never flag anything |
| RespInhib incongruent acc | (0, 1) | Proportion |
| CogFlex | (0, 10) | **Corrected from (0, 150)** — same reason as above |
| Working Memory | (0, 26) | List Sorting raw |
| Attention | (0, 1) | Proportion correct |

---

## Output structure

Column order in the output CSV (35 columns total):

```
participant_id, session_id, site
IQ_raw_notz, IQ_scaled_notz
  IQ_raw_z_withinsite,           IQ_scaled_z_withinsite
  IQ_raw_z_withinsitewave,       IQ_scaled_z_withinsitewave
  IQ_raw_z_withincohort,         IQ_scaled_z_withincohort
  IQ_raw_z_withincohortwave,     IQ_scaled_z_withincohortwave
RespInhib_computed_notz, RespInhib_incongr_acc_notz
  RespInhib_z_withinsite,        RespInhib_z_withinsitewave
  RespInhib_z_withincohort,      RespInhib_z_withincohortwave
  task_type
CogFlex_computed_notz
  CogFlex_z_withinsite,          CogFlex_z_withinsitewave
  CogFlex_z_withincohort,        CogFlex_z_withincohortwave
WorkingMem_raw_notz
  WorkingMem_z_withinsite,       WorkingMem_z_withinsitewave
  WorkingMem_z_withincohort,     WorkingMem_z_withincohortwave
Attention_raw_notz
  Attention_z_withinsite,        Attention_z_withinsitewave
  Attention_z_withincohort,      Attention_z_withincohortwave
```

---

## Changelog

| Version | Changes |
|---|---|
| v1 | Within-site and within-site × wave z-scoring for all 5 domains |
| v2 | Added `_withincohort` and `_withincohortwave` z-scoring for all 5 domains. Millisecond Flanker used mean(accuracy z-score, −1 × median RT z-score) |
| v3 | Millisecond Flanker z-score changed to overall accuracy only. `RespInhib_medrt_notz` removed from output. Fixed column names: `nc_y_nihtb__flnkr__computed_score`, `nc_y_nihtb__crdst__computed_score` (double underscores verified against ABCD v7 data) |
| v4 | Millisecond Flanker z-score changed from overall accuracy to **incongruent accuracy only** (`nc_y_flnkr__incongr_acc`). Congruent trials excluded because they do not test response inhibition. Output column renamed `RespInhib_acc_notz` → `RespInhib_incongr_acc_notz`. LMT data quality filter added: rows with `nc_y_lmt__crct_acc` > 1 (percentage coding errors) set to `n/a`. Column names verified against ABCD v7 data: `nc_y_nihtb__flnkr__computed_score`, `nc_y_nihtb__crdst__computed_score` (double underscores). |
