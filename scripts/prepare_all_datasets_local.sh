#!/usr/bin/env bash

set -u

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DATA_ROOT="${1:-$ROOT_DIR/dataset}"
LOG_FILE="$ROOT_DIR/dataset_setup.log"

mkdir -p "$DATA_ROOT"
mkdir -p "$DATA_ROOT/.tmp"

echo "[INFO] Data root: $DATA_ROOT" | tee "$LOG_FILE"

has_cmd() {
  command -v "$1" >/dev/null 2>&1
}

download_http() {
  local url="$1"
  local out="$2"
  if [[ -f "$out" ]]; then
    echo "[SKIP] Exists: $out" | tee -a "$LOG_FILE"
    return 0
  fi
  echo "[GET ] $url -> $out" | tee -a "$LOG_FILE"
  if has_cmd curl; then
    curl -L --fail --retry 3 -C - "$url" -o "$out" || echo "[WARN] Failed: $url" | tee -a "$LOG_FILE"
  elif has_cmd wget; then
    wget -c -O "$out" "$url" || echo "[WARN] Failed: $url" | tee -a "$LOG_FILE"
  else
    echo "[WARN] Neither curl nor wget found." | tee -a "$LOG_FILE"
  fi
}

extract_if_needed() {
  local archive="$1"
  local target_dir="$2"
  if [[ ! -f "$archive" ]]; then
    echo "[WARN] Archive not found for extraction: $archive" | tee -a "$LOG_FILE"
    return 0
  fi

  mkdir -p "$target_dir"
  case "$archive" in
    *.tar.gz|*.tgz)
      tar -xzf "$archive" -C "$target_dir" || echo "[WARN] Extract failed: $archive" | tee -a "$LOG_FILE"
      ;;
    *.zip)
      if has_cmd unzip; then
        unzip -o "$archive" -d "$target_dir" >/dev/null || echo "[WARN] Extract failed: $archive" | tee -a "$LOG_FILE"
      else
        echo "[WARN] unzip missing, cannot extract: $archive" | tee -a "$LOG_FILE"
      fi
      ;;
    *)
      echo "[WARN] Unknown archive format: $archive" | tee -a "$LOG_FILE"
      ;;
  esac
}

download_gdrive() {
  local url="$1"
  local out="$2"
  if [[ -f "$out" ]]; then
    echo "[SKIP] Exists: $out" | tee -a "$LOG_FILE"
    return 0
  fi

  if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
    "$ROOT_DIR/.venv/bin/python" -m pip show gdown >/dev/null 2>&1 || "$ROOT_DIR/.venv/bin/python" -m pip install gdown >/dev/null 2>&1
    "$ROOT_DIR/.venv/bin/python" -m gdown --fuzzy "$url" -O "$out" || echo "[WARN] gdown failed: $url" | tee -a "$LOG_FILE"
  else
    echo "[WARN] .venv python not found; skipping gdrive download: $url" | tee -a "$LOG_FILE"
  fi
}

echo "[INFO] Creating dataset directories under $DATA_ROOT" | tee -a "$LOG_FILE"
mkdir -p \
  "$DATA_ROOT/imagenet/images" \
  "$DATA_ROOT/caltech-101" \
  "$DATA_ROOT/oxford_pets" \
  "$DATA_ROOT/stanford_cars" \
  "$DATA_ROOT/oxford_flowers" \
  "$DATA_ROOT/food-101" \
  "$DATA_ROOT/fgvc_aircraft" \
  "$DATA_ROOT/sun397" \
  "$DATA_ROOT/dtd" \
  "$DATA_ROOT/eurosat" \
  "$DATA_ROOT/ucf101" \
  "$DATA_ROOT/imagenetv2" \
  "$DATA_ROOT/imagenet-sketch" \
  "$DATA_ROOT/imagenet-adversarial" \
  "$DATA_ROOT/imagenet-rendition"

echo "[INFO] Downloading dataset archives (public links)" | tee -a "$LOG_FILE"

download_http "https://data.caltech.edu/records/mzrjq-6wc02/files/101_ObjectCategories.tar.gz?download=1" "$DATA_ROOT/caltech-101/101_ObjectCategories.tar.gz"
extract_if_needed "$DATA_ROOT/caltech-101/101_ObjectCategories.tar.gz" "$DATA_ROOT/caltech-101"

download_http "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz" "$DATA_ROOT/oxford_pets/images.tar.gz"
download_http "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz" "$DATA_ROOT/oxford_pets/annotations.tar.gz"
extract_if_needed "$DATA_ROOT/oxford_pets/images.tar.gz" "$DATA_ROOT/oxford_pets"
extract_if_needed "$DATA_ROOT/oxford_pets/annotations.tar.gz" "$DATA_ROOT/oxford_pets"

download_http "http://ai.stanford.edu/~jkrause/car196/cars_test.tgz" "$DATA_ROOT/stanford_cars/cars_test.tgz"
download_http "http://ai.stanford.edu/~jkrause/car196/cars_test_annos_withlabels.mat" "$DATA_ROOT/stanford_cars/cars_test_annos_withlabels.mat"
download_http "http://ai.stanford.edu/~jkrause/car196/devkit.tgz" "$DATA_ROOT/stanford_cars/devkit.tgz"
extract_if_needed "$DATA_ROOT/stanford_cars/cars_test.tgz" "$DATA_ROOT/stanford_cars"
extract_if_needed "$DATA_ROOT/stanford_cars/devkit.tgz" "$DATA_ROOT/stanford_cars"

download_http "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz" "$DATA_ROOT/oxford_flowers/102flowers.tgz"
download_http "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat" "$DATA_ROOT/oxford_flowers/imagelabels.mat"
extract_if_needed "$DATA_ROOT/oxford_flowers/102flowers.tgz" "$DATA_ROOT/oxford_flowers"

download_http "https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101.tar.gz" "$DATA_ROOT/food-101/food-101.tar.gz"
extract_if_needed "$DATA_ROOT/food-101/food-101.tar.gz" "$DATA_ROOT"

download_http "https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz" "$DATA_ROOT/fgvc_aircraft/fgvc-aircraft-2013b.tar.gz"
extract_if_needed "$DATA_ROOT/fgvc_aircraft/fgvc-aircraft-2013b.tar.gz" "$DATA_ROOT/fgvc_aircraft"

download_http "http://vision.princeton.edu/projects/2010/SUN/SUN397.tar.gz" "$DATA_ROOT/sun397/SUN397.tar.gz"
download_http "https://vision.princeton.edu/projects/2010/SUN/download/Partitions.zip" "$DATA_ROOT/sun397/Partitions.zip"
extract_if_needed "$DATA_ROOT/sun397/SUN397.tar.gz" "$DATA_ROOT/sun397"
extract_if_needed "$DATA_ROOT/sun397/Partitions.zip" "$DATA_ROOT/sun397"

download_http "https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz" "$DATA_ROOT/dtd/dtd-r1.0.1.tar.gz"
extract_if_needed "$DATA_ROOT/dtd/dtd-r1.0.1.tar.gz" "$DATA_ROOT"

download_http "http://madm.dfki.de/files/sentinel/EuroSAT.zip" "$DATA_ROOT/eurosat/EuroSAT.zip"
extract_if_needed "$DATA_ROOT/eurosat/EuroSAT.zip" "$DATA_ROOT/eurosat"

download_http "https://s3-us-west-2.amazonaws.com/imagenetv2public/imagenetv2-matched-frequency.tar.gz" "$DATA_ROOT/imagenetv2/imagenetv2-matched-frequency.tar.gz"
extract_if_needed "$DATA_ROOT/imagenetv2/imagenetv2-matched-frequency.tar.gz" "$DATA_ROOT/imagenetv2"

echo "[INFO] Downloading split/meta files from Google Drive" | tee -a "$LOG_FILE"

download_gdrive "https://drive.google.com/file/d/1hyarUivQE36mY6jSomru6Fjd-JzwcCzN/view?usp=sharing" "$DATA_ROOT/caltech-101/split_zhou_Caltech101.json"
download_gdrive "https://drive.google.com/file/d/1501r8Ber4nNKvmlFVQZ8SeUHTcdTTEqs/view?usp=sharing" "$DATA_ROOT/oxford_pets/split_zhou_OxfordPets.json"
download_gdrive "https://drive.google.com/file/d/1ObCFbaAgVu0I-k_Au-gIUcefirdAuizT/view?usp=sharing" "$DATA_ROOT/stanford_cars/split_zhou_StanfordCars.json"
download_gdrive "https://drive.google.com/file/d/1AkcxCXeK_RCGCEC_GvmWxjcjaNhu-at0/view?usp=sharing" "$DATA_ROOT/oxford_flowers/cat_to_name.json"
download_gdrive "https://drive.google.com/file/d/1Pp0sRXzZFZq15zVOzKjKBu4A9i01nozT/view?usp=sharing" "$DATA_ROOT/oxford_flowers/split_zhou_OxfordFlowers.json"
download_gdrive "https://drive.google.com/file/d/1QK0tGi096I0Ba6kggatX1ee6dJFIcEJl/view?usp=sharing" "$DATA_ROOT/food-101/split_zhou_Food101.json"
download_gdrive "https://drive.google.com/file/d/1y2RD81BYuiyvebdN-JymPfyWYcd8_MUq/view?usp=sharing" "$DATA_ROOT/sun397/split_zhou_SUN397.json"
download_gdrive "https://drive.google.com/file/d/1u3_QfB467jqHgNXC00UIzbLZRQCg2S7x/view?usp=sharing" "$DATA_ROOT/dtd/split_zhou_DescribableTextures.json"
download_gdrive "https://drive.google.com/file/d/1Ip7yaCWFi0eaOFUGga0lUdVi_DDQth1o/view?usp=sharing" "$DATA_ROOT/eurosat/split_zhou_EuroSAT.json"
download_gdrive "https://drive.google.com/file/d/10Jqome3vtUA2keJkNanAiFpgbyC9Hc2O/view?usp=sharing" "$DATA_ROOT/ucf101/UCF-101-midframes.zip"
download_gdrive "https://drive.google.com/file/d/1I0S0q91hJfsV9Gf4xDIjgDq4AqBNJb1y/view?usp=sharing" "$DATA_ROOT/ucf101/split_zhou_UCF101.json"
download_gdrive "https://drive.google.com/file/d/1-61f_ol79pViBFDG_IDlUQSwoLcn2XXF/view?usp=sharing" "$DATA_ROOT/imagenet/classnames.txt"

extract_if_needed "$DATA_ROOT/ucf101/UCF-101-midframes.zip" "$DATA_ROOT/ucf101"

echo "[NOTE] Manual datasets still required:" | tee -a "$LOG_FILE"
echo "  - ImageNet val + devkit (license-required) under $DATA_ROOT/imagenet/images" | tee -a "$LOG_FILE"
echo "  - ImageNet-Sketch, ImageNet-A, ImageNet-R from their official sources" | tee -a "$LOG_FILE"
echo "  - Copy classnames.txt from $DATA_ROOT/imagenet/classnames.txt into imagenetv2/imagenet-sketch/imagenet-adversarial/imagenet-rendition as needed" | tee -a "$LOG_FILE"

echo "[DONE] Preparation attempt complete. See $LOG_FILE for details." | tee -a "$LOG_FILE"
