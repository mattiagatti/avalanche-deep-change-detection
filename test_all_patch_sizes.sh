#!/bin/bash

# --- Parse arguments ---
USE_AUX=0
MODEL_NAME="swinunet"   # default

for arg in "$@"; do
  if [ "$arg" == "--use-aux" ]; then
    USE_AUX=1
  elif [[ "$arg" == --model=* ]]; then
    MODEL_NAME="${arg#--model=}"
  fi
done

# Patch sizes and GPU mapping
declare -A PATCH_GPU_MAP=(
  [32]=0
  [64]=1
  [128]=2
)

SCRIPT="test.py"
SESSION_NAME="patch_test"

# Create the main tmux session
tmux new-session -d -s "$SESSION_NAME"

FIRST=1
for PATCH_SIZE in "${!PATCH_GPU_MAP[@]}"; do
  GPU_ID=${PATCH_GPU_MAP[$PATCH_SIZE]}
  
  # Base command (now includes model)
  CMD="echo 'Running test: model ${MODEL_NAME}, patch size ${PATCH_SIZE} on GPU ${GPU_ID}'; \
CUDA_VISIBLE_DEVICES=${GPU_ID} python ${SCRIPT} --model \"${MODEL_NAME}\" --patch-size ${PATCH_SIZE}"

  # Append --use-aux if requested
  if [ "$USE_AUX" -eq 1 ]; then
    CMD="$CMD --use-aux"
  fi

  # Finish command
  CMD="$CMD; echo '===== Model ${MODEL_NAME} | Patch ${PATCH_SIZE} Test Complete ====='; bash"

  echo "Launching test for model ${MODEL_NAME}, patch size ${PATCH_SIZE} on GPU ${GPU_ID}"

  if [ "$FIRST" -eq 1 ]; then
    tmux send-keys -t "$SESSION_NAME" "$CMD" C-m
    FIRST=0
  else
    tmux split-window -t "$SESSION_NAME" -h
    tmux send-keys -t "$SESSION_NAME" "$CMD" C-m
    tmux select-layout -t "$SESSION_NAME" tiled
  fi
done

# Attach to the session for monitoring
tmux attach -t "$SESSION_NAME"