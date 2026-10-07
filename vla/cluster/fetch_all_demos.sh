#!/bin/bash
# Download all 10 libero_object demo files (fast on the cluster network).
# Run from $WORK (the directory containing the LIBERO clone).
set -u
DEST=${1:-LIBERO/libero/datasets/libero_object}
mkdir -p "$DEST"
BASE="https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets/resolve/main/libero_object"
for obj in alphabet_soup cream_cheese salad_dressing bbq_sauce ketchup \
           tomato_sauce butter milk chocolate_pudding orange_juice; do
  f="pick_up_the_${obj}_and_place_it_in_the_basket_demo.hdf5"
  curl -L --retry 5 -C - -o "$DEST/$f" "$BASE/$f"
done
echo "ALL DEMOS DONE"
