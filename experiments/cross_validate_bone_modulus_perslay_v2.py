#!/usr/bin/env python3
"""Compatibility entry point for the grouped bone-modulus experiment."""

from cross_validate_bone_modulus_perslay import (
    DIMENSION_SETS,
    fold_assignments,
    main,
    make_loaders,
    make_outer_folds,
    subset_cache,
)


if __name__ == "__main__":
    raise SystemExit(main())
