"""
check_data_integrity.py
------------------------
Standalone diagnostic script covering Task 11 (data distribution), Task 13
(data leakage / pipeline error checks), and the patient-level leakage /
duplicate-group investigation requested for this dataset specifically.

Reports, using the ACTUAL files on disk (never assumed):
  - image counts per split/class (train/val/test x NORMAL/PNEUMONIA)
  - corrupted/unreadable image files (via src.preprocessing.validate_image)
  - ALL duplicate groups (byte-identical files, via MD5), both within a
    single split and across splits, with cross-split groups flagged
    separately as the leakage-relevant subset
  - patient-level leakage: for PNEUMONIA filenames following the
    `person<ID>_...` convention, which patient IDs appear in more than one
    split (the model could partly "recognize" a patient's anatomy/imaging
    style it already saw in training). NORMAL filenames are also scanned,
    and reported honestly as unmatched if they don't follow an extractable
    per-patient naming convention — nothing is assumed about a pattern that
    isn't actually there.

This does NOT catch near-duplicates (e.g. the same X-ray re-exported at a
different resolution/quality) — only byte-identical files. If you suspect
near-duplicates, a perceptual-hash (pHash) pass would be a reasonable
follow-up but is out of scope here since it requires an extra dependency
and a similarity threshold decision.

This script only ever REPORTS findings — it never deletes or moves files.
Any dataset changes based on these findings should be made deliberately and
separately, after reviewing what's actually in the report.

Usage:
    python check_data_integrity.py
"""

from __future__ import annotations

from pathlib import Path

from config import CONFIG, ensure_directories
from src.preprocessing import validate_image
from src.utils import analyze_patient_leakage, count_images, find_all_duplicate_groups, get_logger, save_json


def scan_for_corrupted(directory: Path) -> list:
    directory = Path(directory)
    valid_ext = {".jpg", ".jpeg", ".png", ".bmp"}
    bad = []
    if not directory.exists():
        return bad
    for f in directory.rglob("*"):
        if f.suffix.lower() in valid_ext and not validate_image(f):
            bad.append(str(f))
    return bad


def main():
    ensure_directories()
    logger = get_logger("check_data_integrity", CONFIG.paths.logs_dir)

    split_dirs = {
        "train": CONFIG.paths.train_dir,
        "val": CONFIG.paths.val_dir,
        "test": CONFIG.paths.test_dir,
    }

    # --- Task 11: dataset distribution, from the actual files on disk ---
    print("=" * 72)
    print("DATASET DISTRIBUTION (counted from files on disk)")
    print("=" * 72)
    distribution = {}
    for split_name, split_dir in split_dirs.items():
        counts = count_images(split_dir)
        distribution[split_name] = counts
        total = sum(counts.values())
        counts_str = ", ".join(f"{k}={v}" for k, v in counts.items())
        print(f"{split_name:6s}: {counts_str} | total={total}")

    if not any(distribution.values()):
        print("\n⚠ No images found under dataset/. Download the dataset before running this check.")

    # --- Task 13: corrupted / unreadable images ---
    print("\n" + "=" * 72)
    print("CORRUPTED / UNREADABLE IMAGE SCAN")
    print("=" * 72)
    corrupted = {}
    for split_name, split_dir in split_dirs.items():
        bad = scan_for_corrupted(split_dir)
        corrupted[split_name] = bad
        print(f"{split_name:6s}: {len(bad)} unreadable file(s)")
        for f in bad[:10]:
            print(f"    {f}")

    # --- Duplicate groups: full picture (within-split + cross-split) ---
    print("\n" + "=" * 72)
    print("DUPLICATE GROUP SCAN (byte-identical files only, via MD5)")
    print("=" * 72)
    all_groups = find_all_duplicate_groups(split_dirs)
    cross_split_groups = {h: g for h, g in all_groups.items() if g["spans_multiple_splits"]}
    total_images_in_groups = sum(len(g["locations"]) for g in all_groups.values())

    print(f"Total duplicate groups: {len(all_groups)}  |  images involved: {total_images_in_groups}")
    print(f"Of those, groups crossing a split boundary (real leakage risk): {len(cross_split_groups)}")
    if cross_split_groups:
        print("\nCross-split duplicate groups:")
        for digest, g in list(cross_split_groups.items())[:20]:
            print(f"  hash={digest[:10]}...: {g['locations']}")
        if len(cross_split_groups) > 20:
            print(f"  ... and {len(cross_split_groups) - 20} more (see saved JSON for the full list).")
        print(
            "\nThese are exact byte-for-byte duplicates spanning splits — genuine leakage. "
            "Whether these are truly 'the same image twice' or e.g. two different frames from the same "
            "study that happen to be byte-identical exports, only inspecting the actual files can tell you; "
            "this script deliberately does not delete or move anything."
        )
    else:
        print("No duplicate group crosses a split boundary.")

    # --- Patient-level leakage ---
    print("\n" + "=" * 72)
    print("PATIENT-LEVEL LEAKAGE SCAN (person<ID>_... filename convention)")
    print("=" * 72)
    patient_report = analyze_patient_leakage(split_dirs)
    print(f"Unique patient IDs per split: {patient_report['unique_patient_ids_per_split']}")
    print(f"Filenames with no extractable patient ID per split: {patient_report['unmatched_filenames_per_split']}")
    print("(A high unmatched count in a split is expected if most of that split's images are NORMAL — "
          "NORMAL filenames typically don't follow the person<ID>_ convention at all in this dataset; "
          "this is reported as a fact, not assumed away.)")
    for pair, info in patient_report["overlaps"].items():
        if info["count"] > 0:
            print(f"\n⚠ {info['count']} patient ID(s) appear in BOTH {pair}: {info['patient_ids'][:20]}"
                  + (" ..." if info["count"] > 20 else ""))
        else:
            print(f"\n{pair}: no overlapping patient IDs.")

    if any(info["count"] > 0 for info in patient_report["overlaps"].values()):
        print(
            "\nPatient-level leakage means the same person's X-rays are split across train and "
            "val/test — the model can partly learn that specific patient's anatomy/imaging "
            "characteristics during training and then get an inflated validation/test score on their "
            "other images, rather than genuinely generalizing. This inflates whichever metric you use "
            "to pick a model or threshold. Fixing it means re-splitting by patient ID (grouping all of a "
            "patient's images into a single split), which this script does NOT do automatically."
        )

    # --- Save everything for the record ---
    save_json(
        {
            "distribution": distribution,
            "corrupted_files": corrupted,
            "all_duplicate_groups": all_groups,
            "cross_split_duplicate_groups": cross_split_groups,
            "patient_leakage": patient_report,
        },
        CONFIG.paths.reports_dir / "data_integrity_report.json",
    )
    print(f"\nFull report saved to {CONFIG.paths.reports_dir / 'data_integrity_report.json'}")


if __name__ == "__main__":
    main()
