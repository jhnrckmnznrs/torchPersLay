#!/usr/bin/env python3
"""
Merge baseline, uniform, and weakened morphometry CSVs into one benchmark table.

Default input layout
--------------------
data/
├── morphometry_baseline.csv
├── morphometry_uniform.csv
└── morphometry_weakened.csv

Default output
--------------
data/morphometry_all.csv

The merged table has one row per (structure_id, variant), e.g.

    structure_id,variant,tb_th_mean_mm,...
    structure_0000,baseline,...
    structure_0000,uniform,...
    structure_0000,weakened,...

Notes
-----
The source headers label thickness/spacing as "µm" and BV/TV volumes as
"micron^3", but the numerical values correspond to millimetres for these
synthetic volumes:

  * specimen TV is about 997 = 9.99^3 mm^3
  * trabecular thicknesses around 0.24--0.40 match the strut diameters in mm

Therefore the normalized output column names use mm, mm^3, and mm^-3.

By default the source morphometry BV/TV column is NOT included. The downstream
benchmark should use the exact FEA `retained_density_bvtv` via `--include-bvtv`.
Use --include-source-bvtv only if you explicitly want the morphometry-derived
BV/TV stored as a separate diagnostic column.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


VARIANT_ORDER = ("baseline", "uniform", "weakened")

COLUMN_MAP = {
    "Tb.Th Mean (µm)": "tb_th_mean_mm",
    "Tb.Th Std Dev (µm)": "tb_th_std_mm",
    "Tb.Th Max (µm)": "tb_th_max_mm",
    "Tb.Sp Mean (µm)": "tb_sp_mean_mm",
    "Tb.Sp Std Dev (µm)": "tb_sp_std_mm",
    "Tb.Sp Max (µm)": "tb_sp_max_mm",
    "BV (micron^3)": "bv_mm3",
    "TV (micron^3)": "tv_mm3",
    "Euler χ (ROI-clipped)": "euler_chi_roi_clipped",
    "Connectivity β1 (ROI-clipped)": "connectivity_beta1_roi_clipped",
    "Connectivity Density (µm⁻³)": "connectivity_density_mm3",
}

REQUIRED_SOURCE_COLUMNS = {
    "filename",
    *COLUMN_MAP.keys(),
    "BV/TV",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge morphometry_baseline.csv, morphometry_uniform.csv, and "
            "morphometry_weakened.csv into morphometry_all.csv."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Directory containing the three morphometry CSVs (default: data).",
    )
    parser.add_argument(
        "--baseline",
        default="morphometry_baseline.csv",
        help="Baseline CSV filename relative to --data-root.",
    )
    parser.add_argument(
        "--uniform",
        default="morphometry_uniform.csv",
        help="Uniform CSV filename relative to --data-root.",
    )
    parser.add_argument(
        "--weakened",
        default="morphometry_weakened.csv",
        help="Weakened CSV filename relative to --data-root.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV. Default: <data-root>/morphometry_all.csv",
    )
    parser.add_argument(
        "--include-source-bvtv",
        action="store_true",
        help=(
            "Also retain the morphometry CSV's BV/TV as "
            "`morphometry_bvtv`. By default it is omitted because the "
            "benchmark should use FEA `retained_density_bvtv`."
        ),
    )
    parser.add_argument(
        "--keep-filename",
        action="store_true",
        help="Also keep the original TIFF filename as `source_filename`.",
    )
    return parser.parse_args()


def structure_id_from_filename(value: object) -> str:
    match = re.search(r"(structure_\d+)", str(value), flags=re.IGNORECASE)
    if match is None:
        raise ValueError(
            f"Could not extract structure_XXXX identifier from filename {value!r}"
        )
    return match.group(1).lower()


def structure_number(structure_id: str) -> int:
    return int(structure_id.rsplit("_", 1)[1])


def load_variant(
    path: Path,
    variant: str,
    *,
    include_source_bvtv: bool,
    keep_filename: bool,
) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"{variant}: input CSV not found: {path}")

    df = pd.read_csv(path)

    missing = sorted(REQUIRED_SOURCE_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(
            f"{variant}: {path} is missing required column(s): {missing}"
        )

    out = pd.DataFrame()
    out["structure_id"] = df["filename"].map(structure_id_from_filename)
    out["variant"] = variant

    if keep_filename:
        out["source_filename"] = df["filename"].astype(str)

    for source, normalized in COLUMN_MAP.items():
        out[normalized] = pd.to_numeric(df[source], errors="raise")

    if include_source_bvtv:
        out["morphometry_bvtv"] = pd.to_numeric(df["BV/TV"], errors="raise")

    duplicated = out["structure_id"].duplicated(keep=False)
    if duplicated.any():
        ids = sorted(out.loc[duplicated, "structure_id"].unique())
        raise ValueError(
            f"{variant}: duplicate structure IDs found: {ids[:10]}"
        )

    numeric_columns = out.select_dtypes(include="number").columns
    if out[numeric_columns].isna().any().any():
        bad = out[numeric_columns].columns[
            out[numeric_columns].isna().any()
        ].tolist()
        raise ValueError(f"{variant}: missing numeric values in columns: {bad}")

    return out


def main() -> int:
    args = parse_args()
    data_root = args.data_root

    inputs = {
        "baseline": data_root / args.baseline,
        "uniform": data_root / args.uniform,
        "weakened": data_root / args.weakened,
    }

    frames = {
        variant: load_variant(
            path,
            variant,
            include_source_bvtv=args.include_source_bvtv,
            keep_filename=args.keep_filename,
        )
        for variant, path in inputs.items()
    }

    # Require the three sibling families to contain exactly the same structures.
    reference = set(frames["baseline"]["structure_id"])
    for variant in ("uniform", "weakened"):
        current = set(frames[variant]["structure_id"])
        missing = sorted(reference - current)
        extra = sorted(current - reference)
        if missing or extra:
            raise ValueError(
                f"Structure mismatch for {variant}. "
                f"Missing relative to baseline: {missing[:10]}; "
                f"extra: {extra[:10]}"
            )

    merged = pd.concat(
        [frames[v] for v in VARIANT_ORDER],
        ignore_index=True,
    )

    if merged.duplicated(["structure_id", "variant"]).any():
        raise ValueError("Duplicate (structure_id, variant) rows after merge.")

    # Sort as structure_0000 baseline/uniform/weakened, structure_0001 ..., etc.
    variant_rank = {name: i for i, name in enumerate(VARIANT_ORDER)}
    merged["_structure_number"] = merged["structure_id"].map(structure_number)
    merged["_variant_rank"] = merged["variant"].map(variant_rank)
    merged = (
        merged.sort_values(
            ["_structure_number", "_variant_rank"],
            kind="stable",
        )
        .drop(columns=["_structure_number", "_variant_rank"])
        .reset_index(drop=True)
    )

    output = args.output or (data_root / "morphometry_all.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output, index=False)

    n_structures = merged["structure_id"].nunique()
    print(f"Wrote: {output}")
    print(f"Rows: {len(merged)}")
    print(f"Independent structures: {n_structures}")
    print("Rows per variant:")
    print(merged["variant"].value_counts().reindex(VARIANT_ORDER).to_string())
    print("\nColumns:")
    for column in merged.columns:
        print(f"  {column}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
