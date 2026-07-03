import argparse
import copy
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from perslay.data import (
    MultiPersistenceDiagramDataset,
    PersistenceDiagramDataset,
    collate_multi_persistence_diagrams,
    collate_persistence_diagrams,
    compute_diagram_bounds,
    compute_sigma_from_bounds,
)
from perslay.models import MultiPerslayRegressor, PerslayRegressor
from torch.utils.data import DataLoader, random_split


def move_to_device(x, device):
    if isinstance(x, dict):
        return {key: value.to(device) for key, value in x.items()}

    return x.to(device)


def serialize_image_bnds(image_bnds):
    """
    Convert image bounds to plain Python floats so they can be saved in YAML/checkpoints.

    Supports:
      single diagram:
        [[birth_min, birth_max], [pers_min, pers_max]]

      multi diagram:
        {
          "h0": [[birth_min, birth_max], [pers_min, pers_max]],
          "h2": [[birth_min, birth_max], [pers_min, pers_max]],
        }
    """
    if isinstance(image_bnds, dict):
        return {
            name: [
                [float(bounds[0][0]), float(bounds[0][1])],
                [float(bounds[1][0]), float(bounds[1][1])],
            ]
            for name, bounds in image_bnds.items()
        }

    return [
        [float(image_bnds[0][0]), float(image_bnds[0][1])],
        [float(image_bnds[1][0]), float(image_bnds[1][1])],
    ]


def get_learned_sigmas(model):
    """
    Return learned Gaussian sigmas.

    Single PersLay model:
        (sigma_x, sigma_y)

    Multi PersLay model:
        {
          "h0": (sigma_x, sigma_y),
          "h2": (sigma_x, sigma_y),
        }
    """
    if hasattr(model, "branches"):
        learned = {}

        for name, branch in model.branches.items():
            if hasattr(branch, "phi") and hasattr(branch.phi, "log_sigma"):
                sigma = torch.exp(branch.phi.log_sigma).detach().cpu()
                learned[name] = (
                    float(sigma[0].item()),
                    float(sigma[1].item()),
                )

        return learned if len(learned) > 0 else None

    if hasattr(model, "perslay") and hasattr(model.perslay.phi, "log_sigma"):
        sigma = torch.exp(model.perslay.phi.log_sigma).detach().cpu()
        return float(sigma[0].item()), float(sigma[1].item())

    return None


def make_resolved_config(config, image_bnds, sigma_x_init, sigma_y_init, model):
    """
    Create a config-like dictionary with resolved preprocessing/model values.

    Supports both:
      - single-diagram PersLay
      - multi-diagram PersLay, e.g. H0 + H2
    """
    resolved_config = copy.deepcopy(config)

    resolved_config["model"]["image_bnds"] = serialize_image_bnds(image_bnds)

    learned_sigmas = get_learned_sigmas(model)

    if learned_sigmas is None:
        resolved_config["model"]["sigma_source"] = "not_applicable"
        return resolved_config

    if isinstance(learned_sigmas, dict):
        resolved_config["model"]["learned_sigmas"] = {
            name: {
                "sigma_x": sigma_x,
                "sigma_y": sigma_y,
            }
            for name, (sigma_x, sigma_y) in learned_sigmas.items()
        }

        resolved_config["model"]["initial_sigmas"] = {}

        if isinstance(sigma_x_init, dict) and isinstance(sigma_y_init, dict):
            for name in learned_sigmas.keys():
                resolved_config["model"]["initial_sigmas"][name] = {
                    "sigma_x": float(sigma_x_init[name]),
                    "sigma_y": float(sigma_y_init[name]),
                }

        resolved_config["model"]["sigma_source"] = (
            "learned_from_best_validation_checkpoint_per_branch"
        )

    else:
        learned_sigma_x, learned_sigma_y = learned_sigmas

        resolved_config["model"]["sigma_x"] = learned_sigma_x
        resolved_config["model"]["sigma_y"] = learned_sigma_y
        resolved_config["model"]["initial_sigma_x"] = float(sigma_x_init)
        resolved_config["model"]["initial_sigma_y"] = float(sigma_y_init)
        resolved_config["model"]["sigma_source"] = (
            "learned_from_best_validation_checkpoint"
        )

    return resolved_config


def train_one_epoch(
    model,
    loader,
    optimizer,
    loss_fn,
    device,
    target_mean=None,
    target_std=None,
):
    model.train()

    total_loss = 0.0
    total_count = 0

    for diagrams, mask, targets, filenames in loader:
        diagrams = move_to_device(diagrams, device)
        mask = move_to_device(mask, device)
        targets = targets.to(device)

        if target_mean is not None and target_std is not None:
            target_mean_device = target_mean.to(device)
            target_std_device = target_std.to(device)
            targets_for_loss = (targets - target_mean_device) / target_std_device
        else:
            targets_for_loss = targets

        optimizer.zero_grad()

        preds = model(diagrams, mask=mask)
        loss = loss_fn(preds, targets_for_loss)

        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        batch_size = targets.shape[0]
        total_loss += loss.item() * batch_size
        total_count += batch_size

    return total_loss / total_count


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()

    total_loss = 0.0
    total_count = 0

    for diagrams, mask, targets, filenames in loader:
        diagrams = move_to_device(diagrams, device)
        mask = move_to_device(mask, device)
        targets = targets.to(device)

        preds = model(diagrams, mask=mask)
        loss = loss_fn(preds, targets)

        if isinstance(diagrams, dict):
            first_diagram = next(iter(diagrams.values()))
            batch_size = first_diagram.shape[0]
        else:
            batch_size = diagrams.shape[0]
        total_loss += loss.item() * batch_size
        total_count += batch_size

    return total_loss / total_count


@torch.no_grad()
def evaluate_regression_metrics(
    model, loader, device, target_mean=None, target_std=None
):
    model.eval()

    all_preds = []
    all_targets = []

    for diagrams, mask, targets, filenames in loader:
        diagrams = move_to_device(diagrams, device)
        mask = move_to_device(mask, device)
        targets = targets.to(device)

        preds = model(diagrams, mask=mask)

        # If model predicts standardized targets, convert back to raw scale.
        if target_mean is not None and target_std is not None:
            preds = preds * target_std.to(device) + target_mean.to(device)

        all_preds.append(preds.detach().cpu())
        all_targets.append(targets.detach().cpu())

    y_pred = torch.cat(all_preds)
    y_true = torch.cat(all_targets)

    mse = torch.mean((y_pred - y_true) ** 2)
    rmse = torch.sqrt(mse)
    mae = torch.mean(torch.abs(y_pred - y_true))

    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - y_true.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot.clamp_min(1e-12)

    return {
        "mse": mse.item(),
        "rmse": rmse.item(),
        "mae": mae.item(),
        "r2": r2.item(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/configs/degree2_regression.yaml",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    run_dir = Path(config["output"]["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if "diagram_dirs" in config["data"]:
        dataset = MultiPersistenceDiagramDataset(
            diagram_dirs=config["data"]["diagram_dirs"],
            targets_csv=config["data"]["targets_csv"],
        )
        collate_fn = collate_multi_persistence_diagrams
        is_multi_diagram = True
    else:
        dataset = PersistenceDiagramDataset(
            diagram_dir=config["data"]["diagram_dir"],
            targets_csv=config["data"]["targets_csv"],
        )
        collate_fn = collate_persistence_diagrams
        is_multi_diagram = False

    n_total = len(dataset)
    n_train = int(0.8 * n_total)
    n_val = n_total - n_train

    train_dataset, val_dataset = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    def get_dataset_filename(dataset, idx):
        file_item = dataset.files[idx]

        if hasattr(file_item, "name"):
            return file_item.name

        return str(file_item)

    train_targets = torch.tensor(
        [
            dataset.targets[get_dataset_filename(dataset, idx)]
            for idx in train_dataset.indices
        ],
        dtype=torch.float32,
    )

    target_mean = train_targets.mean()
    target_std = train_targets.std(unbiased=False).clamp_min(1e-8)

    print("Target standardization:")
    print(f"  mean: {target_mean.item():.6f}")
    print(f"  std:  {target_std.item():.6f}")

    if config["model"]["image_bnds"] == "auto":
        if is_multi_diagram:
            image_bnds = {}

            for name in config["data"]["diagram_dirs"].keys():
                image_bnds[name] = compute_diagram_bounds(
                    train_dataset,
                    padding_fraction=config["model"].get(
                        "bounds_padding_fraction", 0.05
                    ),
                    diagram_key=name,
                )

                print(f"Automatically computed image bounds for {name}:")
                print(
                    f"  birth:       "
                    f"[{image_bnds[name][0][0]:.6f}, {image_bnds[name][0][1]:.6f}]"
                )
                print(
                    f"  persistence: "
                    f"[{image_bnds[name][1][0]:.6f}, {image_bnds[name][1][1]:.6f}]"
                )
        else:
            image_bnds = compute_diagram_bounds(
                train_dataset,
                padding_fraction=config["model"].get("bounds_padding_fraction", 0.05),
            )

            print("Automatically computed image bounds from training set:")
            print(f"  birth:       [{image_bnds[0][0]:.6f}, {image_bnds[0][1]:.6f}]")
            print(f"  persistence: [{image_bnds[1][0]:.6f}, {image_bnds[1][1]:.6f}]")
    else:
        image_bnds = config["model"]["image_bnds"]

    model_type = config["model"].get("model_type", "perslay_image")

    if model_type == "perslay_image":
        auto_sigma_x, auto_sigma_y = compute_sigma_from_bounds(
            image_bnds=image_bnds,
            image_size=config["model"]["image_size"],
            multiplier=config["model"].get("sigma_multiplier", 1.0),
        )

        sigma_x_config = config["model"]["sigma_x"]
        sigma_y_config = config["model"]["sigma_y"]

        sigma_x = auto_sigma_x if sigma_x_config == "auto" else float(sigma_x_config)
        sigma_y = auto_sigma_y if sigma_y_config == "auto" else float(sigma_y_config)

        print("Initial Gaussian bandwidths:")
        print(f"  sigma_x: {sigma_x:.6f}")
        print(f"  sigma_y: {sigma_y:.6f}")
    else:
        sigma_x = None
        sigma_y = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config["training"]["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config["training"]["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )

    model_type = config["model"].get("model_type", "perslay_image")

    if model_type == "perslay_image":
        model = PerslayRegressor(
            image_size=config["model"]["image_size"],
            image_bnds=image_bnds,
            sigma_x=sigma_x,
            sigma_y=sigma_y,
            hidden_dim=config["model"]["hidden_dim"],
            weight_type=config["model"].get("weight_type", "learnable_power"),
            weight_hidden_dim=config["model"].get("weight_hidden_dim", 16),
        ).to(device)

    elif model_type == "multi_perslay_image":
        branch_configs = {}

        sigma_x = {}
        sigma_y = {}

        for name in config["data"]["diagram_dirs"].keys():
            image_size = (
                config["model"]
                .get("image_sizes", {})
                .get(
                    name,
                    config["model"]["image_size"],
                )
            )

            branch_image_bnds = image_bnds[name]

            auto_sigma_x, auto_sigma_y = compute_sigma_from_bounds(
                image_bnds=branch_image_bnds,
                image_size=image_size,
                multiplier=config["model"].get("sigma_multiplier", 1.0),
            )

            sigma_x_config = config["model"]["sigma_x"]
            sigma_y_config = config["model"]["sigma_y"]

            sigma_x_branch = (
                auto_sigma_x if sigma_x_config == "auto" else float(sigma_x_config)
            )
            sigma_y_branch = (
                auto_sigma_y if sigma_y_config == "auto" else float(sigma_y_config)
            )

            sigma_x[name] = sigma_x_branch
            sigma_y[name] = sigma_y_branch

            print(f"Initial Gaussian bandwidths for {name}:")
            print(f"  sigma_x: {sigma_x_branch:.6f}")
            print(f"  sigma_y: {sigma_y_branch:.6f}")

            branch_configs[name] = {
                "image_size": image_size,
                "image_bnds": branch_image_bnds,
                "sigma_x": sigma_x_branch,
                "sigma_y": sigma_y_branch,
            }

        model = MultiPerslayRegressor(
            branch_configs=branch_configs,
            hidden_dim=config["model"]["hidden_dim"],
            weight_type=config["model"].get("weight_type", "learnable_power"),
            weight_hidden_dim=config["model"].get("weight_hidden_dim", 16),
            dropout=config["model"].get("dropout", 0.0),
        ).to(device)

    else:
        raise ValueError(
            f"Unknown model_type: {model_type}. "
            "Supported values are 'perslay_image' and 'multi_perslay_image'."
        )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    resume_from = config["training"].get("resume_from")

    if resume_from is not None:
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

        if config["training"].get("resume_optimizer", False):
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        print(f"Loaded model weights from: {resume_from}")
        print(f"Checkpoint epoch: {checkpoint.get('epoch')}")

    loss_fn = nn.MSELoss()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=10,
    )

    best_val_loss = float("inf")

    for epoch in range(config["training"]["epochs"]):
        _ = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            target_mean=target_mean,
            target_std=target_std,
        )

        train_metrics = evaluate_regression_metrics(
            model,
            train_loader,
            device,
            target_mean=target_mean,
            target_std=target_std,
        )

        val_metrics = evaluate_regression_metrics(
            model,
            val_loader,
            device,
            target_mean=target_mean,
            target_std=target_std,
        )

        scheduler.step(val_metrics["mse"])

        print(
            f"epoch={epoch:03d} "
            f"train_mse={train_metrics['mse']:.6f} "
            f"val_mse={val_metrics['mse']:.6f} "
            f"train_rmse={train_metrics['rmse']:.4f} "
            f"val_rmse={val_metrics['rmse']:.4f} "
            f"train_mae={train_metrics['mae']:.4f} "
            f"val_mae={val_metrics['mae']:.4f} "
            f"train_r2={train_metrics['r2']:.4f} "
            f"val_r2={val_metrics['r2']:.4f}"
        )

        current_val_loss = val_metrics["mse"]

        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss

            best_config = make_resolved_config(
                config=config,
                image_bnds=image_bnds,
                sigma_x_init=sigma_x,
                sigma_y_init=sigma_y,
                model=model,
            )

            checkpoint_path = run_dir / "best_model.pt"
            best_config_path = run_dir / "best_config.yaml"

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": config,
                    "resolved_config": best_config,
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                    "train_metrics": train_metrics,
                    "target_mean": target_mean.item(),
                    "target_std": target_std.item(),
                },
                checkpoint_path,
            )

            with open(best_config_path, "w") as f:
                yaml.safe_dump(best_config, f, sort_keys=False)
    print(f"Best validation MSE: {best_val_loss:.6f}")

    if hasattr(model, "perslay") and hasattr(model.perslay.phi, "log_sigma"):
        learned_sigma = torch.exp(model.perslay.phi.log_sigma).detach().cpu()
        print(f"Learned sigma_x: {learned_sigma[0].item():.6f}")
        print(f"Learned sigma_y: {learned_sigma[1].item():.6f}")
    else:
        print("No Gaussian sigma parameters for this model.")



if __name__ == "__main__":
    main()
