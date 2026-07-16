#!/usr/bin/env bash

# Resolve paths relative to this script so it works from any cwd.
cd "$(dirname "$0")"

PATCH_SIZES=(256 128 64 32)
SCRIPT="patchify.py"

# Resample all rasters to a 10 m grid when --to-10m is supplied.
# (patchify.py exposes this as --force-resolution <meters>.)
EXTRA_ARGS=""
if [[ "$1" == "--to-10m" ]]; then
    EXTRA_ARGS="--force-resolution 10"
fi

for PATCH_SIZE in "${PATCH_SIZES[@]}"; do
    STRIDE=$((PATCH_SIZE / 2))

    echo "======================================================="
    echo "Running patchify for patch size $PATCH_SIZE (stride $STRIDE)"
    echo "======================================================="

    python "$SCRIPT" \
        --patch-size "$PATCH_SIZE" \
        --stride "$STRIDE" \
        $EXTRA_ARGS

    echo "Completed: patch size $PATCH_SIZE"
    echo
done

echo "All patchify runs completed."