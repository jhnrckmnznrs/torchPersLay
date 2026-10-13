#!/usr/bin/env python3
"""Repeated grouped PersLay CV for localized-damage targets.

Modes
-----
baseline
    Use the baseline persistence diagram(s) to predict a structure-level target.
    Recommended target: Delta_i_loc. Scientific question: can the original
    architecture predict susceptibility to localized weakening?

paired
    Use shared-weight PersLay encoders on the uniform and weakened diagrams.
    For each homology dimension, the same PersLay map embeds both siblings and
    the regressor receives [U, W, W-U, |W-U|]. Recommended target: Delta_i_loc.

Targets
-------
delta
    Read filename,Delta_i_loc from --delta-csv.

modulus_difference
    Compute E_uniform - E_weakened from the two FEA CSVs.

All outer splits are by structure_id. There is exactly one learning target per
independent structure, so 100 structures remain 100 observations.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

try:
    import train_bone_modulus_perslay as core
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import train_bone_modulus_perslay as core
try:
    import cross_validate_bone_modulus_perslay as cv
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import cross_validate_bone_modulus_perslay as cv


DIMENSION_SETS = cv.DIMENSION_SETS
SPLITS = ("train", "validation", "test")


def read_delta_targets(path: Path) -> dict[str, float]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "Delta_i_loc" not in reader.fieldnames:
            raise core.ExperimentError(f"{path} must contain Delta_i_loc.")
        filename_col = "filename" if "filename" in reader.fieldnames else "relative_image" if "relative_image" in reader.fieldnames else None
        if filename_col is None:
            raise core.ExperimentError(f"{path} must contain filename or relative_image.")
        out: dict[str, float] = {}
        for row in reader:
            sid = core.structure_id_from_filename(str(row[filename_col]))
            value = core.finite_float(row["Delta_i_loc"], context=f"{path}:{sid}")
            if sid in out:
                raise core.ExperimentError(f"Duplicate target for {sid} in {path}.")
            out[sid] = value
    return out


def read_fea_modulus(path: Path, column: str) -> dict[str, float]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        core.require_columns(reader.fieldnames, ("relative_image", "status", "converged", column), path)
        out: dict[str, float] = {}
        for row in reader:
            if str(row["status"]).strip().lower() != "success" or not core.parse_bool(row["converged"]):
                continue
            sid = core.structure_id_from_filename(str(row["relative_image"]))
            out[sid] = core.finite_float(row[column], context=f"{path}:{sid}")
    return out


def read_targets(args: argparse.Namespace) -> dict[str, float]:
    if args.target == "delta":
        return read_delta_targets(args.delta_csv)
    u = read_fea_modulus(args.data_root / "fea_results_uniform_ds3.csv", args.modulus_column)
    w = read_fea_modulus(args.data_root / "fea_results_weakened_ds3.csv", args.modulus_column)
    ids = sorted(set(u) & set(w))
    return {sid: u[sid] - w[sid] for sid in ids}


def make_records(args: argparse.Namespace, dimensions: Sequence[int]) -> list[core.SampleRecord]:
    targets = read_targets(args)
    records: list[core.SampleRecord] = []
    for sid in sorted(targets):
        paths: dict[str, Path] = {}
        if args.mode == "baseline":
            for dim in dimensions:
                name = core.dimension_name(dim)
                p = args.data_root / "diagrams" / f"{name}_baseline" / f"{sid}.csv"
                if not p.is_file():
                    raise core.ExperimentError(f"Missing diagram: {p}")
                paths[name] = p.resolve()
        else:
            for dim in dimensions:
                name = core.dimension_name(dim)
                for variant in ("uniform", "weakened"):
                    key = f"{name}_{variant}"
                    p = args.data_root / "diagrams" / key / f"{sid}.csv"
                    if not p.is_file():
                        raise core.ExperimentError(f"Missing diagram: {p}")
                    paths[key] = p.resolve()
        records.append(
            core.SampleRecord(
                index=len(records),
                sample_id=f"{args.mode}__{sid}",
                structure_id=sid,
                variant=args.mode,
                source_filename=f"{sid}.tif",
                target=float(targets[sid]),
                bvtv=1.0,
                diagram_paths=paths,
            )
        )
    if len(records) < 10:
        raise core.ExperimentError(f"Only {len(records)} structures found.")
    return records


def pooled_pair_settings(cached, assignments, dimensions, padding_fraction):
    bounds = {}
    scales = {}
    for dim in dimensions:
        d = core.dimension_name(dim)
        names = [f"{d}_uniform", f"{d}_weakened"]
        birth_min = math.inf; birth_max = -math.inf
        pers_min = math.inf; pers_max = -math.inf
        masses = []
        found = False
        for idx, split in assignments.items():
            if split != "train":
                continue
            for name in names:
                points, counts = cached[idx][name]
                masses.append(float(counts.sum()))
                if len(points) == 0:
                    continue
                persistence = points[:, 1] - points[:, 0]
                birth_min = min(birth_min, float(points[:, 0].min()))
                birth_max = max(birth_max, float(points[:, 0].max()))
                pers_min = min(pers_min, float(persistence.min()))
                pers_max = max(pers_max, float(persistence.max()))
                found = True
        if not found:
            raise core.ExperimentError(f"All training diagrams empty for {d} pair.")
        br = max(birth_max - birth_min, 1e-6)
        pr = max(pers_max - pers_min, 1e-6)
        bounds[d] = [
            [birth_min - padding_fraction * br, birth_max + padding_fraction * br],
            [max(0.0, pers_min - padding_fraction * pr), pers_max + padding_fraction * pr],
        ]
        scales[d] = max(float(np.mean(masses)), 1.0)
    return bounds, scales


def make_paired_model(*, dimensions, bounds, scales, args, dependencies):
    torch = dependencies.torch
    nn = dependencies.nn

    class SharedPairedPerslay(nn.Module):
        def __init__(self):
            super().__init__()
            self.dimensions = tuple(dimensions)
            branches = {}
            feature_dim = 0
            for dim in self.dimensions:
                d = core.dimension_name(dim)
                sx, sy = core.sigma_from_bounds(bounds[d], args.image_size, multiplier=args.sigma_multiplier)
                wrapper = dependencies.PerslayRegressor(
                    image_size=list(args.image_size), image_bnds=bounds[d],
                    sigma_x=sx, sigma_y=sy, hidden_dim=args.hidden_dimension,
                    weight_type=args.weight_type, weight_hidden_dim=args.weight_hidden_dimension,
                )
                branches[d] = wrapper.perslay
                fd = int(np.prod(wrapper.perslay.phi.mu.shape[1:]))
                feature_dim += 4 * fd
                self.register_buffer(f"mass_scale__{d}", torch.tensor(float(scales[d]), dtype=torch.float32))
            self.branches = nn.ModuleDict(branches)
            self.regressor = nn.Sequential(
                nn.Linear(feature_dim, args.hidden_dimension), nn.ReLU(inplace=True),
                nn.Dropout(args.dropout), nn.Linear(args.hidden_dimension, 1),
            )

        @staticmethod
        def embed(branch, diagrams, multiplicities, mask, mass_scale):
            vector, _ = branch.phi(diagrams)
            point_weight = branch.weight(diagrams)
            point_weight = point_weight * (multiplicities.to(point_weight.dtype) / mass_scale)
            point_weight = point_weight * mask.to(point_weight.dtype)
            for _ in range(vector.ndim - point_weight.ndim):
                point_weight = point_weight.unsqueeze(-1)
            vector = torch.sum(vector * point_weight, dim=1)
            return branch.rho(vector)

        def forward(self, diagrams, multiplicities, masks):
            feats = []
            for dim in self.dimensions:
                d = core.dimension_name(dim)
                branch = self.branches[d]
                scale = getattr(self, f"mass_scale__{d}")
                u = self.embed(branch, diagrams[f"{d}_uniform"], multiplicities[f"{d}_uniform"], masks[f"{d}_uniform"], scale)
                w = self.embed(branch, diagrams[f"{d}_weakened"], multiplicities[f"{d}_weakened"], masks[f"{d}_weakened"], scale)
                feats.extend([u, w, w - u, torch.abs(w - u)])
            return self.regressor(torch.cat(feats, dim=1)).squeeze(-1)

    return SharedPairedPerslay()


def evaluate(model, loader, records, norm, device, torch):
    model.eval(); rows = []
    with torch.no_grad():
        for diagrams, multiplicities, masks, _, _, record_ids in loader:
            diagrams = core.move_mapping(diagrams, device)
            multiplicities = core.move_mapping(multiplicities, device)
            masks = core.move_mapping(masks, device)
            pred = model(diagrams, multiplicities, masks) * norm.std + norm.mean
            for rid, value in zip(record_ids.tolist(), pred.detach().cpu().tolist()):
                idx = int(rid)
                rows.append((idx, records[idx].target, float(value)))
    return rows


def train_fold(records, cached, assignments, split_indices, dimensions, args, dependencies, device, seed):
    torch = dependencies.torch
    core.seed_everything(seed, torch)
    if args.mode == "baseline":
        branch_names = [core.dimension_name(d) for d in dimensions]
        subcache = cv.subset_cache(cached, branch_names)
        bounds = core.compute_training_bounds(subcache, assignments, branch_names, padding_fraction=args.bounds_padding_fraction)
        scales = core.compute_training_mass_scales(subcache, assignments, branch_names)
        model, _ = core.make_model(
            branch_names=branch_names, image_bounds=bounds, image_size=args.image_size,
            sigma_multiplier=args.sigma_multiplier, branch_mass_scales=scales,
            hidden_dimension=args.hidden_dimension, dropout=args.dropout,
            weight_type=args.weight_type, weight_hidden_dimension=args.weight_hidden_dimension,
            dependencies=dependencies,
        )
    else:
        branch_names = [f"{core.dimension_name(d)}_{v}" for d in dimensions for v in ("uniform", "weakened")]
        subcache = cv.subset_cache(cached, branch_names)
        bounds, scales = pooled_pair_settings(subcache, assignments, dimensions, args.bounds_padding_fraction)
        model = make_paired_model(dimensions=dimensions, bounds=bounds, scales=scales, args=args, dependencies=dependencies)

    loaders = cv.make_loaders(
        records=records, split_indices=split_indices, cached=subcache,
        dependencies=dependencies, batch_size=args.batch_size,
        num_workers=args.num_workers, device=device,
    )
    train_targets = np.asarray([records[i].target for i in split_indices["train"]], dtype=float)
    std = float(train_targets.std(ddof=0))
    if std <= 0:
        raise core.ExperimentError("Training target has zero variance.")
    norm = core.TargetNormalization(mean=float(train_targets.mean()), std=std)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=args.scheduler_factor, patience=args.scheduler_patience)
    loss_fn = torch.nn.MSELoss()
    best_rmse = math.inf; best_epoch = 0; best_state = None; stale = 0; history = []
    for epoch in range(1, args.epochs + 1):
        model.train(); total = 0.0; n = 0
        for diagrams, multiplicities, masks, targets, _, _ in loaders["train"]:
            diagrams = core.move_mapping(diagrams, device); multiplicities = core.move_mapping(multiplicities, device); masks = core.move_mapping(masks, device)
            targets = targets.to(device)
            y = (targets - norm.mean) / norm.std
            optimizer.zero_grad(set_to_none=True)
            p = model(diagrams, multiplicities, masks)
            loss = loss_fn(p, y)
            loss.backward()
            if args.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step()
            total += float(loss.item()) * len(targets); n += len(targets)
        val_rows = evaluate(model, loaders["validation"], records, norm, device, torch)
        val = core.regression_metrics(val_rows)
        scheduler.step(val.rmse)
        history.append((epoch, total / max(n,1), val.rmse, val.r2))
        if val.rmse < best_rmse - args.minimum_improvement:
            best_rmse = val.rmse; best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}; stale = 0
        else:
            stale += 1
        if args.verbose_epochs and (epoch == 1 or epoch % 10 == 0):
            print(f"      epoch={epoch} train_norm_MSE={total/max(n,1):.5f} val_RMSE={val.rmse:.5f}")
        if stale >= args.patience:
            break
    if best_state is None:
        raise core.ExperimentError("No valid model state.")
    model.load_state_dict(best_state)
    rows = {split: evaluate(model, loaders[split], records, norm, device, torch) for split in SPLITS}
    return model, norm, best_epoch, rows, history


def main() -> int:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--data-root", type=Path, default=Path("data"))
    ap.add_argument("--output-dir", type=Path, default=Path("outputs/damage_perslay"))
    ap.add_argument("--mode", choices=("baseline", "paired"), default="baseline")
    ap.add_argument("--target", choices=("delta", "modulus_difference"), default="delta")
    ap.add_argument("--delta-csv", type=Path, default=Path("data/delta_i_loc_ds3.csv"))
    ap.add_argument("--modulus-column", default="apparent_youngs_modulus_MPa")
    ap.add_argument("--dimension-sets", nargs="+", choices=tuple(DIMENSION_SETS), default=["h0", "h2", "h0_h2"])
    ap.add_argument("--folds", type=int, default=5); ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--inner-validation-fraction", type=float, default=0.15)
    ap.add_argument("--minimum-persistence", type=float, default=0.0)
    ap.add_argument("--image-size", nargs=2, type=int, default=(10,10))
    ap.add_argument("--bounds-padding-fraction", type=float, default=0.05)
    ap.add_argument("--sigma-multiplier", type=float, default=1.0)
    ap.add_argument("--hidden-dimension", type=int, default=128); ap.add_argument("--dropout", type=float, default=0.10)
    ap.add_argument("--weight-type", choices=("constant","learnable_power","normalized_learnable_power","mlp"), default="normalized_learnable_power")
    ap.add_argument("--weight-hidden-dimension", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=4); ap.add_argument("--epochs", type=int, default=300); ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--minimum-improvement", type=float, default=1e-4); ap.add_argument("--learning-rate", type=float, default=5e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4); ap.add_argument("--gradient-clip", type=float, default=5.0)
    ap.add_argument("--scheduler-patience", type=int, default=10); ap.add_argument("--scheduler-factor", type=float, default=0.5)
    ap.add_argument("--random-seed", type=int, default=2026); ap.add_argument("--num-workers", type=int, default=0); ap.add_argument("--device", default="auto")
    ap.add_argument("--cache-dir", type=Path, default=None); ap.add_argument("--rebuild-cache", action="store_true"); ap.add_argument("--verbose-epochs", action="store_true")
    args = ap.parse_args()
    args.data_root = args.data_root.resolve(); args.output_dir = args.output_dir.resolve()
    if args.target == "delta" and not args.delta_csv.is_absolute():
        args.delta_csv = args.delta_csv.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    selected = tuple(dict.fromkeys(args.dimension_sets))
    dimensions_all = tuple(sorted({d for name in selected for d in DIMENSION_SETS[name]}))
    records = make_records(args, dimensions_all)
    structure_ids = sorted(r.structure_id for r in records)
    folds = cv.make_outer_folds(structure_ids, folds=args.folds, repeats=args.repeats, inner_validation_fraction=args.inner_validation_fraction, random_seed=args.random_seed)
    dependencies = core.import_dependencies(); device = core.choose_device(args.device, dependencies.torch)
    cache_dir = args.cache_dir or (args.output_dir / ".diagram_cache")
    cached, cache_summary = core.build_or_load_cache(records, cache_directory=cache_dir, minimum_persistence=args.minimum_persistence, rebuild=args.rebuild_cache)
    print(f"Loaded {len(records)} independent structures; device={device}")

    prediction_rows = []; metric_rows = []; history_rows = []
    mean_target = float(np.mean([r.target for r in records]))
    for definition in folds:
        assignments, split_indices = cv.fold_assignments(records, definition)
        print(f"repeat={definition.repeat+1}/{args.repeats}, fold={definition.fold+1}/{args.folds}")
        # training-mean baseline is training-only per fold
        fold_mean = float(np.mean([records[i].target for i in split_indices["train"]]))
        test_mean_rows = [(i, records[i].target, fold_mean) for i in split_indices["test"]]
        mm = core.regression_metrics(test_mean_rows)
        metric_rows.append({"repeat":definition.repeat,"outer_fold":definition.fold,"model":"mean","dimension_set":"none","n":mm.number_of_samples,"mae":mm.mae,"rmse":mm.rmse,"r2":mm.r2,"best_epoch":""})
        for i, t, p in test_mean_rows:
            prediction_rows.append({"repeat":definition.repeat,"outer_fold":definition.fold,"model":"mean","dimension_set":"none","structure_id":records[i].structure_id,"target":t,"prediction":p})

        for j, name in enumerate(selected):
            dims = DIMENSION_SETS[name]
            seed = args.random_seed + 100000*definition.repeat + 1000*definition.fold + j
            _, _, best_epoch, rows, history = train_fold(records, cached, assignments, split_indices, dims, args, dependencies, device, seed)
            tm = core.regression_metrics(rows["test"])
            metric_rows.append({"repeat":definition.repeat,"outer_fold":definition.fold,"model":f"perslay_{args.mode}_{args.target}","dimension_set":name,"n":tm.number_of_samples,"mae":tm.mae,"rmse":tm.rmse,"r2":tm.r2,"best_epoch":best_epoch})
            for i,t,p in rows["test"]:
                prediction_rows.append({"repeat":definition.repeat,"outer_fold":definition.fold,"model":f"perslay_{args.mode}_{args.target}","dimension_set":name,"structure_id":records[i].structure_id,"target":t,"prediction":p})
            for epoch, train_mse, val_rmse, val_r2 in history:
                history_rows.append({"repeat":definition.repeat,"outer_fold":definition.fold,"dimension_set":name,"epoch":epoch,"train_normalized_mse":train_mse,"validation_rmse":val_rmse,"validation_r2":val_r2})

    def write(path, rows):
        if not rows: return
        with path.open("w",encoding="utf-8",newline="") as h:
            w=csv.DictWriter(h,fieldnames=list(rows[0].keys()),lineterminator="\n"); w.writeheader(); w.writerows(rows)
    write(args.output_dir/"fold_metrics.csv", metric_rows); write(args.output_dir/"predictions.csv", prediction_rows); write(args.output_dir/"training_history.csv", history_rows)

    summary=[]
    for key in sorted({(r["model"],r["dimension_set"]) for r in metric_rows}):
        g=[r for r in metric_rows if (r["model"],r["dimension_set"])==key]
        rms=np.asarray([float(r["rmse"]) for r in g]); r2=np.asarray([float(r["r2"]) for r in g])
        summary.append({"model":key[0],"dimension_set":key[1],"outer_evaluations":len(g),"rmse_mean":float(rms.mean()),"rmse_std":float(rms.std(ddof=1)),"r2_mean":float(r2.mean())})
    write(args.output_dir/"metric_summary.csv", summary)
    (args.output_dir/"config.json").write_text(json.dumps({"mode":args.mode,"target":args.target,"structures":len(records),"cache_summary":cache_summary,"args":{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}},indent=2)+"\n")
    print("\nTest summary:")
    for r in summary:
        print(f"  {r['model']:30s} {r['dimension_set']:8s} RMSE={r['rmse_mean']:.6g} +/- {r['rmse_std']:.6g}; R2={r['r2_mean']:.4f}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
