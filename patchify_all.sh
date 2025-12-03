#!/usr/bin/env bash

PATCH_SIZES=(256 128 64 32)
SCRIPT="patchify.py"

# Pass through --to-10m if supplied
EXTRA_ARGS=""
if [[ "$1" == "--to-10m" ]]; then
    EXTRA_ARGS="--to-10m"
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