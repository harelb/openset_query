#!/usr/bin/env bash
# Set up the SAM3 environment for the openset_query server.
#
#   setup_env.sh            create (or reuse) the venv and install everything
#                           except model weights; non-interactive, safe to re-run
#   setup_env.sh --weights  make sure the model weights are in the Hugging Face
#                           cache (download when allowed); exits non-zero, with
#                           instructions, when they are missing
#   setup_env.sh --check    report what is present (venv, torch + CUDA, imports,
#                           weights) without installing anything
#
# Environment:
#   ADT4_WS      colcon workspace holding src/awesome_dcist_t4 (heracles, spark_dsg
#                and hydra are installed from there)
#   ADT4_ENV     directory of the ADT4 python environments
#   SAM3_ENV     the venv (default: $ADT4_ENV/sam3)
#   ADT4_REPO    default: $ADT4_WS/src/awesome_dcist_t4
#   HYDRA_PYTHON_SRC  directory holding hydra_python/ (default: found under
#                $ADT4_REPO/hydra); only the pure-python image store is used
#   PYTHON       interpreter for a new venv (default: python3.12, else python3)
#   TORCH_INDEX  torch wheel index (default: CUDA 13.0 wheels)
#   SAM3_GIT     SAM3 source (default: the harelb/sam3 fork at a pinned commit)
#
# SAM3 pins numpy<2, so the server runs in its own venv. Weights are never
# bundled: they come from the Hugging Face Hub (facebook/sam3 is gated).
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MODE=install
case "${1:-}" in
  "") ;;
  --weights) MODE=weights ;;
  --check) MODE=check ;;
  -h | --help)
    sed -n '2,25p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit 0
    ;;
  *)
    echo "unknown argument: $1 (see --help)" >&2
    exit 2
    ;;
esac

if [[ -z "${SAM3_ENV:-}" ]]; then
  [[ -n "${ADT4_ENV:-}" ]] || {
    echo "set ADT4_ENV (or SAM3_ENV)" >&2
    exit 1
  }
  SAM3_ENV=$ADT4_ENV/sam3
fi
PY=$SAM3_ENV/bin/python
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu130}
SAM3_GIT=${SAM3_GIT:-git+https://github.com/harelb/sam3.git@a708c3de40c9f652e246b153bf43a62c8f5c0841}
LOCK=$HERE/requirements-lock.txt

# Hugging Face repos and the files the server loads from each:
#   facebook/sam3: SAM3 image model (sam3.model_builder.build_sam3_image_model)
#   google/siglip2-so400m-patch16-naflex: SigLIP2 frame ranking + crop scoring
SAM3_REPO=facebook/sam3
SAM3_FILES="config.json sam3.pt"
SIGLIP_REPO=google/siglip2-so400m-patch16-naflex
SIGLIP_FILES="config.json model.safetensors preprocessor_config.json special_tokens_map.json tokenizer_config.json tokenizer.json"

# --- weights ------------------------------------------------------------------

# cached REPO FILE... -> 0 when every file is in the local HF cache
cached() {
  "$PY" - "$@" 2>/dev/null <<'EOF'
import sys
from huggingface_hub import try_to_load_from_cache
repo, files = sys.argv[1], sys.argv[2:]
sys.exit(0 if all(isinstance(try_to_load_from_cache(repo, f), str) for f in files) else 1)
EOF
}

weights_help() {
  local repo=$1 dir="models--${1/\//--}"
  cat >&2 <<EOF

  $repo is not in the Hugging Face cache (\${HF_HOME:-~/.cache/huggingface}/hub).
  Either:
    a) accept the model license at https://huggingface.co/$repo (gated models
       only), log in, and re-run:
         $SAM3_ENV/bin/hf auth login
         bash $HERE/setup_env.sh --weights
    b) or copy the cache folder from a machine that has it:
         rsync -a <host>:.cache/huggingface/hub/$dir ~/.cache/huggingface/hub/
  Then run the server with HF_HUB_OFFLINE=1.
EOF
}

# fetch REPO FILE... -> 0 when the files are (now) cached
fetch() {
  local repo=$1
  shift
  if cached "$repo" "$@"; then
    echo "  $repo: cached"
    return 0
  fi
  if [[ "${HF_HUB_OFFLINE:-0}" == 1 ]]; then
    echo "  $repo: missing (HF_HUB_OFFLINE=1, not downloading)"
  elif [[ -x "$SAM3_ENV/bin/hf" ]]; then
    echo "  $repo: downloading"
    "$SAM3_ENV/bin/hf" download "$repo" "$@" >/dev/null || true
    if cached "$repo" "$@"; then
      echo "  $repo: cached"
      return 0
    fi
    echo "  $repo: download failed (gated model without access, no token, or offline)"
  else
    echo "  $repo: missing (no hf CLI in $SAM3_ENV; run setup_env.sh first)"
  fi
  weights_help "$repo"
  return 1
}

do_weights() {
  [[ -x "$PY" ]] || {
    echo "no venv at $SAM3_ENV: run setup_env.sh first" >&2
    return 1
  }
  echo "### model weights"
  local ok=0
  # shellcheck disable=SC2086
  fetch "$SAM3_REPO" $SAM3_FILES || ok=1
  # shellcheck disable=SC2086
  fetch "$SIGLIP_REPO" $SIGLIP_FILES || ok=1
  return $ok
}

# --- check --------------------------------------------------------------------

do_check() {
  local ok=0
  if [[ -x "$PY" ]]; then
    echo "venv: $SAM3_ENV ($("$PY" -V 2>&1))"
  else
    echo "venv: MISSING ($SAM3_ENV)"
    return 1
  fi
  "$PY" - <<'EOF' || ok=1
import importlib, sys, warnings
warnings.simplefilter("ignore")
bad = 0
try:
    import torch
    cuda = torch.cuda.is_available()
    dev = torch.cuda.get_device_name(0) if cuda else "-"
    print(f"torch: {torch.__version__}, CUDA available: {cuda} ({dev})")
except Exception as e:  # noqa: BLE001
    print(f"torch: MISSING ({e})")
    bad = 1
for mod in ("sam3", "transformers", "spark_dsg", "heracles",
            "hydra_python.image_store", "openset_query.query_server"):
    try:
        importlib.import_module(mod)
        print(f"import {mod}: ok")
    except Exception as e:  # noqa: BLE001
        print(f"import {mod}: FAILED ({type(e).__name__}: {e})")
        bad = 1
sys.exit(bad)
EOF
  local spec repo
  for spec in "$SAM3_REPO:$SAM3_FILES" "$SIGLIP_REPO:$SIGLIP_FILES"; do
    repo=${spec%%:*}
    # shellcheck disable=SC2086
    if cached "$repo" ${spec#*:}; then
      echo "weights $repo: cached"
    else
      echo "weights $repo: MISSING (setup_env.sh --weights)"
      ok=1
    fi
  done
  return $ok
}

# --- install ------------------------------------------------------------------

find_hydra_python() {
  if [[ -n "${HYDRA_PYTHON_SRC:-}" ]]; then
    echo "$HYDRA_PYTHON_SRC"
    return
  fi
  local d
  for d in "$ADT4_REPO/hydra/hydra/python/src" "$ADT4_REPO/hydra/python/src"; do
    if [[ -d "$d/hydra_python/image_store" ]]; then
      echo "$d"
      return
    fi
  done
}

do_install() {
  [[ -n "${ADT4_WS:-}" || -n "${ADT4_REPO:-}" ]] || {
    echo "set ADT4_WS (or ADT4_REPO)" >&2
    exit 1
  }
  ADT4_REPO=${ADT4_REPO:-$ADT4_WS/src/awesome_dcist_t4}
  local hydra_src d
  hydra_src=$(find_hydra_python)
  for d in "$ADT4_REPO/heracles/heracles" "$ADT4_REPO/spark_dsg" "$hydra_src"; do
    [[ -n "$d" && -d "$d" ]] || {
      echo "missing ADT4 source: ${d:-hydra_python (set HYDRA_PYTHON_SRC)}" >&2
      exit 1
    }
  done

  if [[ -x "$PY" ]]; then
    echo "### SAM3 environment already exists at $SAM3_ENV, reusing it"
  else
    local python=${PYTHON:-}
    if [[ -z "$python" ]]; then
      python=$(command -v python3.12 || command -v python3)
    fi
    echo "### creating SAM3 environment at $SAM3_ENV ($python)"
    mkdir -p "$(dirname "$SAM3_ENV")"
    "$python" -m venv "$SAM3_ENV"
  fi
  local pip=("$PY" -m pip install --disable-pip-version-check)
  "${pip[@]}" --upgrade pip wheel setuptools

  echo "### torch + torchvision ($TORCH_INDEX)"
  "${pip[@]}" torch==2.12.0 torchvision==0.27.0 --index-url "$TORCH_INDEX"

  echo "### SAM3 ($SAM3_GIT) and the deps its import graph needs"
  "${pip[@]}" -c "$LOCK" "sam3 @ $SAM3_GIT"
  # declared optional by SAM3 but needed to import its inference entry points
  "${pip[@]}" -c "$LOCK" einops pycocotools psutil pandas python-rapidjson \
    hydra-core scipy torchmetrics fvcore fairscale scikit-image scikit-learn

  echo "### spark_dsg + heracles from $ADT4_REPO"
  "${pip[@]}" -c "$LOCK" "$ADT4_REPO/spark_dsg"
  "${pip[@]}" -c "$LOCK" -e "$ADT4_REPO/heracles/heracles"

  echo "### hydra_python.image_store from $hydra_src (pure python, path entry)"
  local site
  site=$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
  echo "$hydra_src" >"$site/adt4_hydra_python.pth"

  echo "### openset_query (editable) + the hf CLI"
  "${pip[@]}" -c "$LOCK" -e "$HERE" huggingface_hub "numpy<2"

  echo "### done; next: bash $HERE/setup_env.sh --weights"
}

case $MODE in
  install) do_install ;;
  weights) do_weights ;;
  check) do_check ;;
esac
