# Synthetic apparent-modulus experiments

## Data layout

Run commands from the repository root. The grouped experiment expects:

```text
data/
├── fea_results_baseline_ds3.csv
├── fea_results_uniform_ds3.csv
├── fea_results_weakened_ds3.csv
└── diagrams/
    ├── h0_baseline/
    ├── h0_uniform/
    ├── h0_weakened/
    ├── h2_baseline/
    ├── h2_uniform/
    └── h2_weakened/
```

Each diagram file must be named `structure_XXXX.csv` and contain `birth` and
`death` columns. Each FEA table must contain `relative_image`, `status`,
`converged`, `apparent_youngs_modulus_MPa`, and `retained_density_bvtv`.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[experiments]"
```

## Smoke test

```bash
python experiments/cross_validate_bone_modulus_perslay.py \
  --data-root data \
  --output-dir outputs/feature_map_smoke \
  --variants baseline uniform weakened \
  --folds 2 --repeats 1 --epochs 3 --patience 2 \
  --models variant_residual \
  --dimension-sets h0 \
  --feature-map-types grid_sampled integrated \
  --gaussian-normalization normalized \
  --flatten-order row-major \
  --device auto
```

## Main controlled comparison

This configuration holds Gaussian normalization, vector ordering, point
weighting, folds, and initialization seeds fixed. The intended difference is
centre sampling versus exact pixel integration.

```bash
python experiments/cross_validate_bone_modulus_perslay.py \
  --data-root data \
  --output-dir outputs/bone_modulus_discretization_comparison \
  --variants baseline uniform weakened \
  --folds 5 --repeats 10 \
  --inner-validation-fraction 0.15 \
  --models variant_residual \
  --dimension-sets h0 h2 h0_h2 \
  --feature-map-types grid_sampled integrated \
  --gaussian-normalization normalized \
  --flatten-order row-major \
  --learnable-sigma \
  --minimum-persistence 0.0 \
  --image-size 10 10 \
  --bounds-padding-fraction 0.05 \
  --sigma-multiplier 1.0 \
  --weight-type normalized_learnable_power \
  --hidden-dimension 128 \
  --dropout 0.10 \
  --batch-size 4 \
  --epochs 300 \
  --patience 40 \
  --minimum-improvement 0.0001 \
  --learning-rate 0.0005 \
  --weight-decay 0.0001 \
  --gradient-clip 5.0 \
  --scheduler-patience 10 \
  --scheduler-factor 0.5 \
  --random-seed 2026 \
  --device auto
```

This command performs 300 trainings: 10 repetitions, 5 outer folds, 3
homology-dimension sets, and 2 feature maps.

## Conventional representation defaults

For a sensitivity analysis comparing the conventional forms of the two
representations, use a separate output directory and replace the normalization
and flattening arguments with:

```bash
--gaussian-normalization representation-default \
--flatten-order representation-default
```

The sampled surface is then unnormalized and legacy-x-major, whereas the
integrated image uses normalized pixel masses and row-major ordering.

## Analysis

Run the paired feature-map comparison for each dimension set:

```bash
python experiments/compare_bone_modulus_feature_maps.py \
  --cv-output outputs/bone_modulus_discretization_comparison \
  --model perslay_variant_residual \
  --dimension-set h0_h2 \
  --feature-map-types grid_sampled integrated \
  --bootstrap 5000 \
  --seed 2026
```

Analyze one feature map against the variant-aware BV/TV baseline:

```bash
python experiments/analyze_bone_modulus_cv.py \
  --cv-output outputs/bone_modulus_discretization_comparison \
  --output-dir outputs/bone_modulus_discretization_comparison/post_analysis_integrated \
  --model perslay_variant_residual \
  --dimension-set h0_h2 \
  --feature-map-type integrated \
  --baseline variant_bvtv \
  --bootstrap 5000 \
  --seed 2026
```

Using the same output directory resumes an interrupted experiment. Use a new
directory for an independent rerun. Do not use `--overwrite-output` unless the
selected directory may be deleted deliberately.
