#!/bin/bash
# ===========================================================================
# inspect_abcc_xcpd.sh
#
# Read-only inspection of an ABCC / XCP-D output tree, to determine whether
# corrected connectivity matrices can be recomputed without re-running XCP-D.
#
# The two XCP-D 0.13.0 bugs (per-run instead of concatenated 300/600 matrices,
# and no z-scoring before concatenation) both live AFTER parcellation, so the
# fix only needs parcellated timeseries plus run boundaries. This script works
# out which of those are actually present.
#
# Usage:
#   bash inspect_abcc_xcpd.sh /path/to/xcp_d/sub-XXXXXX
#   bash inspect_abcc_xcpd.sh /path/to/xcp_d/sub-XXXXXX/ses-YYY
#
# Writes nothing. Reads nothing but filenames and .tsv line counts.
# ===========================================================================

set -uo pipefail

ROOT="${1:-}"
if [ -z "$ROOT" ] || [ ! -d "$ROOT" ]; then
    echo "Usage: bash inspect_abcc_xcpd.sh <xcp_d subject or session directory>" >&2
    exit 1
fi

echo "======================================================================"
echo " INSPECTING: $ROOT"
echo "======================================================================"

# --- helper: report a file class -------------------------------------------
# $1 = label, $2.. = find -name patterns (OR'd)
report() {
    local label="$1"; shift
    local -a pats=()
    local p
    for p in "$@"; do pats+=( -o -name "$p" ); done
    unset 'pats[0]'   # drop leading -o

    local files
    files=$(find "$ROOT" -type f \( "${pats[@]}" \) 2>/dev/null | sort)
    local n; n=$(printf '%s\n' "$files" | sed '/^$/d' | wc -l | tr -d '[:space:]')

    printf "\n%s\n  count: %s\n" "--- ${label} ---" "$n"
    if [ "$n" -eq 0 ]; then
        echo "  (none found)"
        return
    fi

    local n_run n_norun
    n_run=$(printf '%s\n'   "$files" | sed '/^$/d' | grep -c '_run-'    || true)
    n_norun=$(printf '%s\n' "$files" | sed '/^$/d' | grep -vc '_run-'   || true)
    echo "  with run- entity:    $n_run"
    echo "  without run- entity: $n_norun"
    echo "  examples:"
    printf '%s\n' "$files" | sed '/^$/d' | head -4 | sed 's|.*/|    |'

    # Line counts for tsv timeseries tell us concatenated vs per-run length.
    local f rows
    while IFS= read -r f; do
        [ -n "$f" ] || continue
        case "$f" in
            *.tsv) rows=$(( $(wc -l < "$f" 2>/dev/null || echo 1) - 1 ))
                   printf "    rows=%-7s %s\n" "$rows" "$(basename "$f")" ;;
        esac
    done < <(printf '%s\n' "$files" | sed '/^$/d' | head -6)
}

report "PARCELLATED TIMESERIES (.tsv)"      '*atlas-*timeseries.tsv'
report "PARCELLATED TIMESERIES (.ptseries)" '*.ptseries.nii'
report "CONNECTIVITY MATRICES"              '*relmat.tsv' '*.pconn.nii'
report "TEMPORAL MASK / OUTLIERS"           '*outliers.tsv' '*temporalmask*.tsv'
report "MOTION"                             '*motion.tsv'
report "DENOISED DENSE TIMESERIES"          '*desc-denoised*bold.dtseries.nii' '*desc-denoised*bold.nii.gz'

# --- desc- values present on matrices (tells us the requested lengths) -----
echo
echo "--- desc- values found on relmat files ---"
find "$ROOT" -type f -name '*relmat*' 2>/dev/null \
    | sed 's|.*/||' | grep -oE 'desc-[A-Za-z0-9]+' | sort | uniq -c \
    | sed 's/^/  /' || echo "  (none)"

# --- TR, if a sidecar exposes it ------------------------------------------
echo
echo "--- RepetitionTime from any JSON sidecar ---"
find "$ROOT" -type f -name '*.json' 2>/dev/null | head -20 \
    | xargs grep -ho '"RepetitionTime"[[:space:]]*:[[:space:]]*[0-9.]*' 2>/dev/null \
    | sort -u | sed 's/^/  /' || echo "  (not found)"

# ===========================================================================
# VERDICT
# ===========================================================================
pts_run=$(find "$ROOT" -type f \( -name '*atlas-*timeseries.tsv' -o -name '*.ptseries.nii' \) 2>/dev/null | grep -c '_run-' || true)
pts_any=$(find "$ROOT" -type f \( -name '*atlas-*timeseries.tsv' -o -name '*.ptseries.nii' \) 2>/dev/null | wc -l | tr -d '[:space:]')
mask_run=$(find "$ROOT" -type f \( -name '*outliers.tsv' -o -name '*temporalmask*.tsv' -o -name '*motion.tsv' \) 2>/dev/null | grep -c '_run-' || true)

echo
echo "======================================================================"
echo " VERDICT"
echo "======================================================================"
if [ "$pts_any" -eq 0 ]; then
    echo "  NO parcellated timeseries found."
    echo "  -> Either this is the wrong directory, or the release ships only"
    echo "     dense timeseries. Check the DENOISED DENSE block above: if those"
    echo "     exist, matrices can still be rebuilt, but you must parcellate"
    echo "     yourself with the same atlases (Gordon/HCP/MIDB)."
elif [ "$pts_run" -gt 0 ]; then
    echo "  PER-RUN parcellated timeseries ARE distributed ($pts_run files)."
    echo "  -> Direct route. Z-score each run, concatenate, correlate."
    echo "     No splitting needed, no re-run of XCP-D."
elif [ "$mask_run" -gt 0 ]; then
    echo "  Only CONCATENATED parcellated timeseries, but per-run temporal"
    echo "  mask / motion files exist ($mask_run files)."
    echo "  -> Split route. Recover run boundaries from the per-run files,"
    echo "     split the concatenated series, z-score each run, re-concatenate."
    echo "     Still no re-run of XCP-D needed."
    echo "  CHECK: does the sum of per-run row counts equal the concatenated"
    echo "  row count? If yes the series is censored consistently and the"
    echo "  boundaries are exact. If not, the concatenated file is probably"
    echo "  interpolated rather than censored - tell me and the split changes."
else
    echo "  Concatenated parcellated timeseries only, and NO per-run motion or"
    echo "  temporal mask files found."
    echo "  -> Run boundaries are not recoverable from this directory alone."
    echo "     Re-running XCP-D 26.x on the fMRIPrep derivatives would be the"
    echo "     reliable route. Check whether ABCC distributes those separately."
fi
echo
