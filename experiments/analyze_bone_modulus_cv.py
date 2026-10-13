#!/usr/bin/env python3
"""Post-analysis for repeated grouped PersLay modulus cross-validation.

Consumes predictions.csv from cross_validate_bone_modulus_perslay.py/v2 and
produces:
  * valid per-variant OOF metrics,
  * uniform-vs-weakened paired E_U - E_W prediction diagnostics,
  * structure-level bootstrap CIs for RMSE improvement,
  * residual diagnostics for the variant-aware BV/TV residual model,
  * diagnostic plots.

The repeated OOF predictions are averaged per sample before structure-level
bootstrap analysis, so the 100 underlying geometries—not 300 siblings or 50
fold evaluations—are treated as the independent units.

When one CV output contains several PersLay feature maps, the requested map is
selected explicitly before any averaging or metric calculation.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def regression_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    err = p - y
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y - np.mean(y))**2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {"n": int(len(y)), "mae": mae, "rmse": rmse, "r2": r2}


def corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return corr(pd.Series(x).rank().to_numpy(), pd.Series(y).rank().to_numpy())


def poly_r2(x: np.ndarray, y: np.ndarray, degree: int) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) <= degree or np.std(x) == 0:
        return float("nan")
    coeff = np.polyfit(x, y, degree)
    pred = np.polyval(coeff, x)
    return regression_metrics(y, pred)["r2"]


def average_oof(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["sample_id", "structure_id", "variant"]
    agg = {
        "target_MPa": "first",
        "retained_density_bvtv": "first",
        "prediction_MPa": "mean",
    }
    for col in ("actual_log_residual", "predicted_log_residual", "base_log_prediction"):
        if col in df.columns:
            agg[col] = "mean"
    out = df.groupby(group_cols, as_index=False).agg(agg)
    counts = df.groupby(group_cols).size().rename("oof_repeats").reset_index()
    return out.merge(counts, on=group_cols, how="left")


def bootstrap_improvement(
    model_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    *,
    n_boot: int,
    seed: int,
    scope: str,
) -> dict[str, float]:
    left = model_df.copy()
    right = baseline_df[["sample_id", "prediction_MPa"]].rename(
        columns={"prediction_MPa": "baseline_prediction_MPa"}
    )
    merged = left.merge(right, on="sample_id", how="inner", validate="one_to_one")
    if scope != "all":
        merged = merged[merged["variant"] == scope].copy()
    structures = np.asarray(sorted(merged["structure_id"].unique()), dtype=object)
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n_boot):
        sampled = rng.choice(structures, size=len(structures), replace=True)
        pieces = [merged[merged["structure_id"] == sid] for sid in sampled]
        boot = pd.concat(pieces, ignore_index=True)
        y = boot["target_MPa"].to_numpy(float)
        rmse_m = regression_metrics(y, boot["prediction_MPa"].to_numpy(float))["rmse"]
        rmse_b = regression_metrics(y, boot["baseline_prediction_MPa"].to_numpy(float))["rmse"]
        deltas.append(rmse_m - rmse_b)
    a = np.asarray(deltas)
    observed_y = merged["target_MPa"].to_numpy(float)
    observed = (
        regression_metrics(observed_y, merged["prediction_MPa"].to_numpy(float))["rmse"]
        - regression_metrics(observed_y, merged["baseline_prediction_MPa"].to_numpy(float))["rmse"]
    )
    return {
        "scope": scope,
        "n_structures": int(len(structures)),
        "observed_delta_rmse_MPa": float(observed),
        "bootstrap_mean_delta_rmse_MPa": float(a.mean()),
        "ci025_delta_rmse_MPa": float(np.quantile(a, 0.025)),
        "ci975_delta_rmse_MPa": float(np.quantile(a, 0.975)),
        "probability_delta_below_zero": float(np.mean(a < 0.0)),
    }


def paired_uniform_weakened(
    avg: pd.DataFrame, label: str
) -> tuple[pd.DataFrame, dict[str, float]]:
    u = avg[avg["variant"] == "uniform"].copy()
    w = avg[avg["variant"] == "weakened"].copy()
    keep = ["structure_id", "target_MPa", "prediction_MPa", "retained_density_bvtv"]
    pair = u[keep].merge(
        w[keep],
        on="structure_id",
        suffixes=("_uniform", "_weakened"),
        validate="one_to_one",
    )
    pair["true_E_uniform_minus_weakened_MPa"] = (
        pair["target_MPa_uniform"] - pair["target_MPa_weakened"]
    )
    pair["pred_E_uniform_minus_weakened_MPa"] = (
        pair["prediction_MPa_uniform"] - pair["prediction_MPa_weakened"]
    )
    pair["bvtv_uniform_minus_weakened"] = (
        pair["retained_density_bvtv_uniform"] - pair["retained_density_bvtv_weakened"]
    )
    y = pair["true_E_uniform_minus_weakened_MPa"].to_numpy(float)
    p = pair["pred_E_uniform_minus_weakened_MPa"].to_numpy(float)
    m = regression_metrics(y, p)
    nonzero = np.abs(y) > 1e-12
    sign_acc = (
        float(np.mean(np.sign(y[nonzero]) == np.sign(p[nonzero])))
        if np.any(nonzero)
        else float("nan")
    )
    summary = {
        "model": label,
        "n_pairs": int(len(pair)),
        "paired_difference_mae_MPa": m["mae"],
        "paired_difference_rmse_MPa": m["rmse"],
        "paired_difference_r2": m["r2"],
        "pearson_true_vs_pred_difference": corr(y, p),
        "spearman_true_vs_pred_difference": spearman(y, p),
        "sign_accuracy": sign_acc,
        "mean_abs_bvtv_pair_mismatch": float(
            np.mean(np.abs(pair["bvtv_uniform_minus_weakened"]))
        ),
    }
    return pair, summary


def bootstrap_paired(pair: pd.DataFrame, n_boot: int, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    y = pair["true_E_uniform_minus_weakened_MPa"].to_numpy(float)
    p = pair["pred_E_uniform_minus_weakened_MPa"].to_numpy(float)
    vals = []
    signs = []
    n = len(pair)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        vals.append(regression_metrics(y[idx], p[idx])["rmse"])
        nz = np.abs(y[idx]) > 1e-12
        signs.append(
            float(np.mean(np.sign(y[idx][nz]) == np.sign(p[idx][nz])))
            if np.any(nz)
            else np.nan
        )
    vals = np.asarray(vals)
    signs = np.asarray(signs)
    return {
        "paired_rmse_ci025_MPa": float(np.quantile(vals, 0.025)),
        "paired_rmse_ci975_MPa": float(np.quantile(vals, 0.975)),
        "sign_accuracy_ci025": float(np.nanquantile(signs, 0.025)),
        "sign_accuracy_ci975": float(np.nanquantile(signs, 0.975)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv-output", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--model", default="perslay_variant_residual")
    ap.add_argument("--dimension-set", default="h0_h2")
    ap.add_argument(
        "--feature-map-type",
        choices=("auto", "legacy_gaussian", "grid_sampled", "integrated"),
        default="auto",
        help=(
            "PersLay feature map to analyze. 'auto' is accepted only when the "
            "selected model and dimension set contain one feature map."
        ),
    )
    ap.add_argument("--baseline", default="variant_bvtv")
    ap.add_argument("--bootstrap", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    pred_path = args.cv_output / "predictions.csv"
    if not pred_path.is_file():
        raise SystemExit(f"Missing {pred_path}")
    out = args.output_dir or (args.cv_output / "post_analysis")
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(pred_path)
    if "feature_map_type" not in df.columns:
        df["feature_map_type"] = np.where(
            df["model"].astype(str).str.startswith("perslay_"),
            "legacy_gaussian",
            "none",
        )
    test = df[df["split"] == "test"].copy()

    model_candidates = test[
        (test["model"] == args.model)
        & (test["dimension_set"] == args.dimension_set)
    ].copy()
    if model_candidates.empty:
        raise SystemExit(
            f"No test rows for model={args.model}, dimension_set={args.dimension_set}"
        )
    available_feature_maps = sorted(
        model_candidates["feature_map_type"].dropna().astype(str).unique().tolist()
    )
    if args.feature_map_type == "auto":
        if len(available_feature_maps) != 1:
            raise SystemExit(
                "The selected model contains multiple feature maps: "
                f"{available_feature_maps}. Supply --feature-map-type explicitly."
            )
        selected_feature_map = available_feature_maps[0]
    else:
        selected_feature_map = args.feature_map_type
    model_rows = model_candidates[
        model_candidates["feature_map_type"] == selected_feature_map
    ].copy()
    base_rows = test[(test["model"] == args.baseline) & (test["dimension_set"] == "none")].copy()
    if model_rows.empty:
        raise SystemExit(
            f"No test rows for model={args.model}, dimension_set="
            f"{args.dimension_set}, feature_map_type={selected_feature_map}"
        )
    if base_rows.empty:
        raise SystemExit(f"No test rows for baseline={args.baseline}")

    # Guard against accidental stale/reused runs.
    for label, x in (("model", model_rows), ("baseline", base_rows)):
        dup = x.duplicated(["repeat", "sample_id"], keep=False)
        if dup.any():
            raise SystemExit(
                f"{label} predictions contain duplicate repeat/sample rows. "
                "Use a clean all-three-variant CV output directory."
            )

    model_avg = average_oof(model_rows)
    base_avg = average_oof(base_rows)
    model_avg.to_csv(out / "oof_average_perslay.csv", index=False)
    base_avg.to_csv(out / "oof_average_variant_bvtv.csv", index=False)

    metric_rows = []
    model_label = (
        args.model + "__" + args.dimension_set + "__" + selected_feature_map
    )
    for label, x in ((model_label, model_avg), (args.baseline, base_avg)):
        for scope in ["all"] + sorted(x["variant"].unique().tolist()):
            s = x if scope == "all" else x[x["variant"] == scope]
            m = regression_metrics(
                s["target_MPa"].to_numpy(float),
                s["prediction_MPa"].to_numpy(float),
            )
            metric_rows.append({"model": label, "variant_scope": scope, **m})
    pd.DataFrame(metric_rows).to_csv(out / "oof_variant_metrics.csv", index=False)

    ci_rows = []
    scopes = ["all"] + sorted(model_avg["variant"].unique().tolist())
    for i, scope in enumerate(scopes):
        ci_rows.append(
            bootstrap_improvement(
                model_avg, base_avg, n_boot=args.bootstrap, seed=args.seed + i, scope=scope
            )
        )
    pd.DataFrame(ci_rows).to_csv(out / "structure_bootstrap_rmse_improvement.csv", index=False)

    pair_model, pair_summary_model = paired_uniform_weakened(model_avg, model_label)
    pair_base, pair_summary_base = paired_uniform_weakened(base_avg, args.baseline)
    pair_model.to_csv(out / "paired_uniform_weakened_perslay.csv", index=False)
    pair_base.to_csv(out / "paired_uniform_weakened_variant_bvtv.csv", index=False)
    pair_summary_model.update(bootstrap_paired(pair_model, args.bootstrap, args.seed + 100))
    pair_summary_base.update(bootstrap_paired(pair_base, args.bootstrap, args.seed + 101))
    pd.DataFrame([pair_summary_model, pair_summary_base]).to_csv(
        out / "paired_uniform_weakened_summary.csv", index=False
    )

    # Residual diagnostics for residual models.
    residual_rows = []
    if "actual_log_residual" in model_avg.columns and np.isfinite(
        model_avg["actual_log_residual"]
    ).any():
        for scope in ["all"] + sorted(model_avg["variant"].unique().tolist()):
            s = model_avg if scope == "all" else model_avg[model_avg["variant"] == scope]
            x = np.log(s["retained_density_bvtv"].to_numpy(float))
            actual = s["actual_log_residual"].to_numpy(float)
            predicted = s["predicted_log_residual"].to_numpy(float)
            residual_rows.append(
                {
                    "variant_scope": scope,
                    "n": len(s),
                    "pearson_actual_vs_predicted_residual": corr(actual, predicted),
                    "spearman_actual_vs_predicted_residual": spearman(actual, predicted),
                    "pearson_actual_residual_vs_log_bvtv": corr(actual, x),
                    "pearson_predicted_residual_vs_log_bvtv": corr(predicted, x),
                    "quadratic_R2_actual_residual_from_log_bvtv": poly_r2(x, actual, 2),
                    "cubic_R2_actual_residual_from_log_bvtv": poly_r2(x, actual, 3),
                }
            )
        pd.DataFrame(residual_rows).to_csv(out / "residual_diagnostics.csv", index=False)

        fig, ax = plt.subplots(figsize=(6, 6))
        for variant, s in model_avg.groupby("variant"):
            ax.scatter(
                s["actual_log_residual"],
                s["predicted_log_residual"],
                alpha=0.7,
                label=variant,
            )
        lo = min(
            model_avg["actual_log_residual"].min(),
            model_avg["predicted_log_residual"].min(),
        )
        hi = max(
            model_avg["actual_log_residual"].max(),
            model_avg["predicted_log_residual"].max(),
        )
        ax.plot([lo, hi], [lo, hi], linestyle="--")
        ax.set_xlabel("Actual log-modulus residual")
        ax.set_ylabel("Predicted log-modulus residual")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "residual_actual_vs_predicted.png", dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 5))
        for variant, s in model_avg.groupby("variant"):
            ax.scatter(
                np.log(s["retained_density_bvtv"]),
                s["actual_log_residual"],
                alpha=0.7,
                label=variant,
            )
        ax.axhline(0.0, linestyle="--")
        ax.set_xlabel("log(BV/TV)")
        ax.set_ylabel("Actual residual after variant-aware BV/TV model")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "actual_residual_vs_log_bvtv.png", dpi=180)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(
        pair_model["true_E_uniform_minus_weakened_MPa"],
        pair_model["pred_E_uniform_minus_weakened_MPa"],
        alpha=0.75,
    )
    lo = min(
        pair_model["true_E_uniform_minus_weakened_MPa"].min(),
        pair_model["pred_E_uniform_minus_weakened_MPa"].min(),
    )
    hi = max(
        pair_model["true_E_uniform_minus_weakened_MPa"].max(),
        pair_model["pred_E_uniform_minus_weakened_MPa"].max(),
    )
    ax.plot([lo, hi], [lo, hi], linestyle="--")
    ax.axhline(0.0, linewidth=0.8)
    ax.axvline(0.0, linewidth=0.8)
    ax.set_xlabel("True E_uniform - E_weakened (MPa)")
    ax.set_ylabel("Predicted E_uniform - E_weakened (MPa)")
    fig.tight_layout()
    fig.savefig(out / "paired_uniform_weakened_difference.png", dpi=180)
    plt.close(fig)

    summary = {
        "model": args.model,
        "dimension_set": args.dimension_set,
        "feature_map_type": selected_feature_map,
        "baseline": args.baseline,
        "bootstrap_replicates": args.bootstrap,
        "output_directory": str(out),
        "note": (
            "Repeated OOF predictions are averaged per sample; bootstrap "
            "resamples structure_id and keeps siblings together."
        ),
    }
    (out / "analysis_config.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    print(f"Wrote post-analysis to {out}")
    print(pd.DataFrame(metric_rows).to_string(index=False))
    print("\nPaired uniform-weakened difference:")
    print(pd.DataFrame([pair_summary_model, pair_summary_base]).to_string(index=False))
    print("\nStructure-bootstrap RMSE improvement (PersLay - variant-aware BV/TV):")
    print(pd.DataFrame(ci_rows).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
