#!/bin/bash
set -u
cd "$(dirname "$0")"
DEST=LIBERO/libero/datasets/libero_object
mkdir -p "$DEST"
BASE="https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets/resolve/main/libero_object"
for f in \
  pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5 \
  pick_up_the_cream_cheese_and_place_it_in_the_basket_demo.hdf5 \
  pick_up_the_salad_dressing_and_place_it_in_the_basket_demo.hdf5 \
  pick_up_the_bbq_sauce_and_place_it_in_the_basket_demo.hdf5 \
  pick_up_the_ketchup_and_place_it_in_the_basket_demo.hdf5; do
  echo "[fetch $(date +%H:%M:%S)] $f"
  curl -L --retry 5 --retry-delay 5 -C - -o "$DEST/$f" "$BASE/$f"
done
echo "SEEN DEMOS DONE $(date +%H:%M:%S)"
