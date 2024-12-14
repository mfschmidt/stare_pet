#!/bin/bash

ERROR=0

MEAN_IMG=$(ls -1 ./*_orig_mean.nii.gz)
if [[ ! -f "$MEAN_IMG" ]]; then
  echo "Cannot find mean image for background"
  ERROR=1
fi

STEP_1_IMG=$(ls -1 ./masks/cluster_step-1_best_mask.nii.gz)
if [[ ! -f "$STEP_1_IMG" ]]; then
  echo "Cannot find step 1 cluster mask"
  ERROR=1
fi

STEP_2_IMG=$(ls -1 ./masks/cluster_step-2_best_mask.nii.gz)
if [[ ! -f "$STEP_2_IMG" ]]; then
  echo "Cannot find step 2 cluster mask"
  ERROR=1
fi

if [[ "$ERROR" == "1" ]]; then
  sys.exit 1
fi

# We have three files; load them into fsleyes.
fsleyes \
"${MEAN_IMG}" \
  --name "Average PET" --overlayType volume \
"${STEP_1_IMG}" \
  --name "K-Means Step 1 Cluster" --overlayType mask \
  --maskColour 1.0 1.0 0.0 --alpha 50 \
"${STEP_2_IMG}" \
  --name "K-Means Step 2 Cluster" --overlayType mask \
  --maskColour 1.0 0.0 0.0 --alpha 50

#   --worldLoc x y z or --voxelLoc x y z \
# Notes:
#
# If fsleyes is older than version 1.10.2 (January, 2024),
# transparency will not be applied to the masks.
# See https://fsl.fmrib.ox.ac.uk/fsl/docs/#/install/index/ to update.
