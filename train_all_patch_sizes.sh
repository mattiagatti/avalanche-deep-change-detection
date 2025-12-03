#!/bin/bash

# --- Parse arguments ---
USE_AUX_FLAG=""
MODEL_NAME="swinunet"   # default

for arg in "$@"; do
  if [ "$arg" = "--use-aux" ]; then
    USE_AUX_FLAG="--use-aux"
  elif [[ "$arg" == --model=* ]]; then
    MODEL_NAME="${arg#--model=}"
  fi
done

declare -A PATCH_GPU_MAP=(
  [32]=1
  [64]=2
  [128]=4
)

SESSION_NAME="patch_training"
tmux new-session -d -s "$SESSION_NAME"

FIRST=1
for PATCH_SIZE in "${!PATCH_GPU_MAP[@]}"; do
  GPU_ID=${PATCH_GPU_MAP[$PATCH_SIZE]}
  DESCRIPTION="training ${MODEL_NAME} patch_size ${PATCH_SIZE}"
  CMD="CUDA_VISIBLE_DEVICES=${GPU_ID} python train.py \
        --model ${MODEL_NAME} \
        --patch-size ${PATCH_SIZE} \
        --description \"${DESCRIPTION}\" \
        $USE_AUX_FLAG"

  echo "Launching: $DESCRIPTION on GPU $GPU_ID"
  if [ "$FIRST" -eq 1 ]; then
    tmux send-keys -t "$SESSION_NAME" "$CMD" C-m
    FIRST=0
  else
    tmux split-window -t "$SESSION_NAME" -h
    tmux send-keys -t "$SESSION_NAME" "$CMD" C-m
    tmux select-layout -t "$SESSION_NAME" tiled
  fi
done

tmux attach -t "$SESSION_NAME"